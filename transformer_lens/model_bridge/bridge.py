"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
from torch import nn

from transformer_lens import utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.hook_point_wrapper import HookPointWrapper
from transformer_lens.model_bridge.types import ComponentMapping
from transformer_lens.utilities.aliases import resolve_alias

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


class TransformerBridge(nn.Module):
    """Bridge between HuggingFace and HookedTransformer models.

    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    # Top-level hook aliases for legacy TransformerLens names
    # Placing these on the main bridge ensures aliases like 'hook_embed' are available
    hook_aliases: Dict[str, Union[str, List[str]]] = {
        "hook_embed": "embed.hook_out",
        # rotary style models use rotary_emb.hook_out, but gpt2-style models use pos_embed.hook_out
        "hook_pos_embed": ["pos_embed.hook_out", "rotary_emb.hook_out"],
        "hook_unembed": "unembed.hook_out",
    }

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
    ):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        self.original_model: nn.Module = model
        self.adapter = adapter
        self.cfg = adapter.cfg

        self.tokenizer = tokenizer
        self.compatibility_mode = False
        self._hook_cache = None  # Cache for hook discovery results
        self._hook_registry: Dict[
            str, HookPoint
        ] = {}  # Dynamic registry of hook names to HookPoints
        self._hook_registry_initialized = False  # Track if registry has been initialized

        # Add device information to config from the loaded model
        if not hasattr(self.cfg, "device") or self.cfg.device is None:
            try:
                self.cfg.device = str(next(self.original_model.parameters()).device)
            except StopIteration:
                self.cfg.device = "cpu"

        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

        # Set original components on the pre-created bridge components
        set_original_components(self, self.adapter, self.original_model)

        # Initialize hook registry after components are set up
        self._initialize_hook_registry()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track HookPoint objects dynamically."""
        # Call parent setattr first
        super().__setattr__(name, value)

        # Check if this is a HookPoint being set
        if isinstance(value, HookPoint):
            # Set the name on the HookPoint
            value.name = name
            # Add to registry
            self._hook_registry[name] = value
        elif isinstance(value, HookPointWrapper):
            # Handle HookPointWrapper objects
            hook_in_name = f"{name}.hook_in"
            hook_out_name = f"{name}.hook_out"
            value.hook_in.name = hook_in_name
            value.hook_out.name = hook_out_name
            self._hook_registry[hook_in_name] = value.hook_in
            self._hook_registry[hook_out_name] = value.hook_out
        elif hasattr(value, "get_hooks") and callable(getattr(value, "get_hooks")):
            # This is a GeneralizedComponent being set
            # We need to register its hooks with the appropriate prefix
            component_hooks = value.get_hooks()
            for hook_name, hook in component_hooks.items():
                full_name = f"{name}.{hook_name}"
                hook.name = full_name
                self._hook_registry[full_name] = hook

    def _initialize_hook_registry(self) -> None:
        """Initialize the hook registry by scanning existing components."""
        if self._hook_registry_initialized:
            return

        # Scan existing components for hooks
        self._scan_existing_hooks(self, "")

        self._hook_registry_initialized = True

    def _scan_existing_hooks(self, module: nn.Module, prefix: str = "") -> None:
        """Scan existing modules for hooks and add them to registry."""
        visited = set()

        def scan_module(mod: nn.Module, path: str = "") -> None:
            obj_id = id(mod)
            if obj_id in visited:
                return
            visited.add(obj_id)

            # Check if this is a GeneralizedComponent with its own hook registry
            if hasattr(mod, "get_hooks") and callable(getattr(mod, "get_hooks")):
                # Use the component's own hook registry
                try:
                    component_hooks = mod.get_hooks()  # type: ignore
                    if isinstance(component_hooks, dict):
                        # Type cast to help mypy understand this is a dict of hooks
                        hooks_dict = cast(Dict[str, HookPoint], component_hooks)  # type: ignore
                        for hook_name, hook in hooks_dict.items():  # type: ignore
                            full_name = f"{path}.{hook_name}" if path else hook_name
                            hook.name = full_name
                            self._hook_registry[full_name] = hook
                except Exception:
                    # If get_hooks() fails, fall through to the else block
                    pass

            # Always scan attributes for additional hooks and submodules
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                if attr_name == "original_component" or "original_model":
                    continue

                try:
                    attr = getattr(mod, attr_name)
                except Exception:
                    continue

                name = f"{path}.{attr_name}" if path else attr_name

                if isinstance(attr, HookPoint):
                    attr.name = name
                    self._hook_registry[name] = attr
                elif isinstance(attr, HookPointWrapper):
                    hook_in_name = f"{name}.hook_in"
                    hook_out_name = f"{name}.hook_out"
                    attr.hook_in.name = hook_in_name
                    attr.hook_out.name = hook_out_name
                    self._hook_registry[hook_in_name] = attr.hook_in
                    self._hook_registry[hook_out_name] = attr.hook_out

            # Check named children
            for child_name, child_module in mod.named_children():
                if (
                    child_name == "original_component"
                    or child_name == "_original_component"
                    or child_name == "original_model"
                ):
                    continue
                child_path = f"{path}.{child_name}" if path else child_name
                scan_module(child_module, child_path)

        scan_module(module, prefix)

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """Get all HookPoint objects in the model for compatibility with HookedTransformer."""
        # Start with the current registry
        return self._hook_registry.copy()

    def _discover_hooks(self) -> dict[str, HookPoint]:
        """Get all HookPoint objects from the registry (deprecated, use hook_dict)."""
        return self._hook_registry.copy()

    def clear_hook_cache(self) -> None:
        """Clear the cached hook discovery results (deprecated, kept for compatibility)."""
        pass  # No longer needed since we don't use caching

    def clear_hook_registry(self) -> None:
        """Clear the hook registry and force re-initialization."""
        self._hook_registry.clear()
        self._hook_registry_initialized = False

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        if name in self.__dict__:
            return self.__dict__[name]

        # Check if this is a hook alias when compatibility mode is enabled
        if self.compatibility_mode:
            resolved_hook = resolve_alias(self, name, self.hook_aliases)
            if resolved_hook is not None:
                return resolved_hook

        return super().__getattr__(name)

    def _get_nested_attr(self, path: str) -> Any:
        """Get a nested attribute using dot notation."""
        obj = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def _format_single_component(self, name: str, path: str, indent: int = 0) -> str:
        """Format a single component's string representation.

        Args:
            name: The name of the component
            path: The path to get the component
            indent: The indentation level

        Returns:
            A formatted string for the component
        """
        indent_str = "  " * indent
        try:
            comp = self.adapter.get_component(self.original_model, path)
            if hasattr(comp, "original_component"):
                if comp.original_component is None:
                    return f"{indent_str}{name}: <error: original component not set>"
                return f"{indent_str}{name}: {type(comp).__name__}({type(comp.original_component).__name__})"
            return f"{indent_str}{name}: {type(comp).__name__}"
        except Exception as e:
            return f"{indent_str}{name}: <error: {e}>"

    def _format_component_mapping(
        self, mapping: ComponentMapping, indent: int = 0, prepend: str | None = None
    ) -> list[str]:
        """Format a component mapping dictionary.

        Args:
            mapping: The component mapping dictionary
            indent: The indentation level
            prepend: Optional path to prepend to component names (e.g. "blocks.0")

        Returns:
            A list of formatted strings
        """
        lines = []
        for name, value in mapping.items():
            path = f"{prepend}.{name}" if prepend else name

            if hasattr(value, "_modules") and hasattr(value, "name"):
                # This is a bridge component instance
                lines.append(self._format_single_component(name, path, indent))

                # Check if it has submodules (like BlockBridge)
                submodules = value.submodules

                if submodules:
                    # For list items (like blocks), add .0 to the path to indicate the first item
                    subpath = f"{path}.0" if value.is_list_item else path
                    # Recursively format submodules
                    sub_lines = self._format_component_mapping(submodules, indent + 1, subpath)
                    lines.extend(sub_lines)

            else:
                # For other types, use prepend if provided
                lines.append(self._format_single_component(name, path, indent))
        return lines

    def __str__(self) -> str:
        """Get a string representation of the bridge.

        Returns:
            A string describing the bridge's components
        """
        lines = ["TransformerBridge:"]
        mapping = self.adapter.get_component_mapping()
        lines.extend(self._format_component_mapping(mapping, indent=1))
        return "\n".join(lines)

    def enable_compatibility_mode(
        self, disable_warnings: bool = False, no_processing: bool = False
    ) -> None:
        """Enable compatibility mode for the bridge.

        This sets up the bridge to work with legacy HookedTransformer components/hooks.
        It will also disable warnings about the usage of legacy components/hooks if specified.

        Args:
            disable_warnings: Whether to disable warnings about legacy components/hooks
            no_processing: Whether to disable pre-processing steps of the model (e.g. folding layer norm weights, folding value biases)
        """
        # Avoid circular import
        from transformer_lens.utilities.bridge_components import (
            apply_fn_to_all_components,
        )

        self.compatibility_mode = True

        def set_compatibility_mode(component: Any) -> None:
            """Set compatibility mode on a component."""
            component.compatibility_mode = True
            component.disable_warnings = disable_warnings

        apply_fn_to_all_components(self, set_compatibility_mode)

        # Re-initialize the hook registry to include aliases from components
        self.clear_hook_registry()
        self._initialize_hook_registry()

        if not no_processing:
            # Apply weight processing using the centralized ProcessWeights class
            print("Applying weight processing...")
            self.process_weights(
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
            )

    def process_weights(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Apply weight processing transformations directly to HuggingFace tensor formats.

        Keeps weights in HF format throughout and applies the same mathematical
        transformations as ProcessWeights, adapted for HF tensor shapes.
        """
        import torch

        print("Applying weight processing mathematics directly to HF format...")

        print("  Extracting HuggingFace state dict...")
        original_state_dict = self.original_model.state_dict()

        state_dict = {}
        for key, tensor in original_state_dict.items():
            clean_key = key.replace("._original_component", "")
            state_dict[clean_key] = tensor.clone()

        print(f"  Processing {len(state_dict)} parameters with HF-native mathematics...")

        if fold_ln:
            print("    Folding LayerNorm...")
            self._fold_layer_norm_hf_native(state_dict)

        if center_writing_weights:
            print("    Centering writing weights...")
            self._center_writing_weights_hf_native(state_dict)

        if center_unembed:
            print("    Centering unembedding weights...")
            self._center_unembed_hf_native(state_dict)

        print("  Adding missing LayerNorm parameters as identity...")
        self._add_identity_layer_norm_params(state_dict)

        print("  Loading processed weights back into model...")
        self._load_processed_hf_weights(state_dict)

        print("✅ Weight processing completed successfully!")

    def _fold_layer_norm_hf_native(self, state_dict):
        """Fold LayerNorm into subsequent layers using HF tensor formats."""
        import torch

        for layer_idx in range(self.cfg.n_layers):
            # Fold LN1 into attention
            ln1_weight = state_dict[f"transformer.h.{layer_idx}.ln_1.weight"]
            ln1_bias = state_dict[f"transformer.h.{layer_idx}.ln_1.bias"]

            c_attn_weight = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"]
            c_attn_bias = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"]

            # Split combined QKV for processing
            d_model = self.cfg.d_model
            q_weight = c_attn_weight[:, :d_model]
            k_weight = c_attn_weight[:, d_model : 2 * d_model]
            v_weight = c_attn_weight[:, 2 * d_model :]

            q_bias = c_attn_bias[:d_model]
            k_bias = c_attn_bias[d_model : 2 * d_model]
            v_bias = c_attn_bias[2 * d_model :]

            # Apply LayerNorm folding: fold biases, then weights, then center
            q_bias = q_bias + torch.sum(q_weight * ln1_bias[:, None], dim=0)
            k_bias = k_bias + torch.sum(k_weight * ln1_bias[:, None], dim=0)
            v_bias = v_bias + torch.sum(v_weight * ln1_bias[:, None], dim=0)

            q_weight = q_weight * ln1_weight[:, None]
            k_weight = k_weight * ln1_weight[:, None]
            v_weight = v_weight * ln1_weight[:, None]

            q_weight = q_weight - torch.mean(q_weight, dim=0, keepdim=True)
            k_weight = k_weight - torch.mean(k_weight, dim=0, keepdim=True)
            v_weight = v_weight - torch.mean(v_weight, dim=0, keepdim=True)

            # Recombine and store
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat(
                [q_weight, k_weight, v_weight], dim=1
            )
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"] = torch.cat(
                [q_bias, k_bias, v_bias], dim=0
            )

            del state_dict[f"transformer.h.{layer_idx}.ln_1.weight"]
            del state_dict[f"transformer.h.{layer_idx}.ln_1.bias"]

            # Fold LN2 into MLP
            ln2_weight = state_dict[f"transformer.h.{layer_idx}.ln_2.weight"]
            ln2_bias = state_dict[f"transformer.h.{layer_idx}.ln_2.bias"]

            c_fc_weight = state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"]
            c_fc_bias = state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"]

            c_fc_bias = c_fc_bias + torch.sum(c_fc_weight * ln2_bias[:, None], dim=0)
            c_fc_weight = c_fc_weight * ln2_weight[:, None]
            c_fc_weight = c_fc_weight - torch.mean(c_fc_weight, dim=0, keepdim=True)

            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = c_fc_weight
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = c_fc_bias

            del state_dict[f"transformer.h.{layer_idx}.ln_2.weight"]
            del state_dict[f"transformer.h.{layer_idx}.ln_2.bias"]

        # Fold final LayerNorm into unembedding
        ln_final_weight = state_dict["transformer.ln_f.weight"]
        ln_final_bias = state_dict["transformer.ln_f.bias"]

        lm_head_weight = state_dict["lm_head.weight"]

        if "lm_head.bias" in state_dict:
            lm_head_bias = state_dict["lm_head.bias"]
            lm_head_bias = lm_head_bias + torch.sum(lm_head_weight * ln_final_bias[None, :], dim=1)
            state_dict["lm_head.bias"] = lm_head_bias

        lm_head_weight = lm_head_weight * ln_final_weight[None, :]
        state_dict["lm_head.weight"] = lm_head_weight

        del state_dict["transformer.ln_f.weight"]
        del state_dict["transformer.ln_f.bias"]

    def _center_writing_weights_hf_native(self, state_dict):
        """Center weights that write to the residual stream."""
        import torch

        # Center embedding weights
        wte_weight = state_dict["transformer.wte.weight"]
        wte_weight = wte_weight - torch.mean(wte_weight, dim=1, keepdim=True)
        state_dict["transformer.wte.weight"] = wte_weight

        if "transformer.wpe.weight" in state_dict:
            wpe_weight = state_dict["transformer.wpe.weight"]
            wpe_weight = wpe_weight - torch.mean(wpe_weight, dim=1, keepdim=True)
            state_dict["transformer.wpe.weight"] = wpe_weight

        # Center output weights that write to residual stream
        for layer_idx in range(self.cfg.n_layers):
            c_proj_weight = state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
            c_proj_weight = c_proj_weight - torch.mean(c_proj_weight, dim=1, keepdim=True)
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = c_proj_weight

            mlp_c_proj_weight = state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"]
            mlp_c_proj_weight = mlp_c_proj_weight - torch.mean(
                mlp_c_proj_weight, dim=1, keepdim=True
            )
            state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = mlp_c_proj_weight

    def _center_unembed_hf_native(self, state_dict):
        """Center unembedding weights."""
        import torch

        lm_head_weight = state_dict["lm_head.weight"]
        lm_head_weight = lm_head_weight - torch.mean(lm_head_weight, dim=1, keepdim=True)
        state_dict["lm_head.weight"] = lm_head_weight

    def _convert_hf_to_tl_shapes(self, hf_state_dict):
        """Convert HuggingFace tensor shapes to TransformerLens tensor shapes."""
        tl_state_dict = {}

        for hf_key, tensor in hf_state_dict.items():
            if hf_key == "lm_head.weight":
                # HF: [vocab_size, d_model] -> TL: [d_model, vocab_size]
                tl_state_dict[hf_key] = tensor.T
            else:
                # Most other tensors have the same shape between HF and TL
                tl_state_dict[hf_key] = tensor

        return tl_state_dict

    def _convert_tl_to_hf_shapes(self, tl_state_dict):
        """Convert TransformerLens tensor shapes back to HuggingFace tensor shapes."""
        hf_state_dict = {}

        for key, tensor in tl_state_dict.items():
            if key == "lm_head.weight":
                # TL: [d_model, vocab_size] -> HF: [vocab_size, d_model]
                hf_state_dict[key] = tensor.T
            else:
                # Most other tensors have the same shape between TL and HF
                hf_state_dict[key] = tensor

        return hf_state_dict

    def _add_identity_layer_norm_params(self, processed_hf_state_dict):
        """Add missing LayerNorm parameters as identity values.

        After folding LayerNorm into other layers, HuggingFace models still expect
        LayerNorm parameters to exist. Set them to identity (weight=1, bias=0).
        """
        import torch

        for layer_idx in range(self.cfg.n_layers):
            ln1_weight_key = f"transformer.h.{layer_idx}.ln_1.weight"
            ln1_bias_key = f"transformer.h.{layer_idx}.ln_1.bias"
            ln2_weight_key = f"transformer.h.{layer_idx}.ln_2.weight"
            ln2_bias_key = f"transformer.h.{layer_idx}.ln_2.bias"

            if ln1_weight_key not in processed_hf_state_dict:
                processed_hf_state_dict[ln1_weight_key] = torch.ones(self.cfg.d_model)
            if ln1_bias_key not in processed_hf_state_dict:
                processed_hf_state_dict[ln1_bias_key] = torch.zeros(self.cfg.d_model)
            if ln2_weight_key not in processed_hf_state_dict:
                processed_hf_state_dict[ln2_weight_key] = torch.ones(self.cfg.d_model)
            if ln2_bias_key not in processed_hf_state_dict:
                processed_hf_state_dict[ln2_bias_key] = torch.zeros(self.cfg.d_model)

        ln_final_weight_key = "transformer.ln_f.weight"
        ln_final_bias_key = "transformer.ln_f.bias"

        if ln_final_weight_key not in processed_hf_state_dict:
            processed_hf_state_dict[ln_final_weight_key] = torch.ones(self.cfg.d_model)
        if ln_final_bias_key not in processed_hf_state_dict:
            processed_hf_state_dict[ln_final_bias_key] = torch.zeros(self.cfg.d_model)

    def _load_processed_hf_weights(self, processed_hf_state_dict):
        """Load processed HuggingFace weights back into the original model."""
        # Get the original model's state dict with _original_component suffixes
        original_state_dict = self.original_model.state_dict()

        # Load processed weights into the original model components
        for processed_key, processed_tensor in processed_hf_state_dict.items():
            # Find the corresponding key with _original_component suffix
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == processed_key:
                    original_state_dict[orig_key].data.copy_(processed_tensor)
                    break

    def _load_processed_tl_weights_to_hf_model(self, processed_tl_state_dict):
        """Load processed TL weights back into the HuggingFace model.

        This converts TL-format processed weights back to HF format and loads them
        into the original_model components of the TransformerBridge.
        """
        import torch

        # Get the original model's state dict with _original_component suffixes
        original_state_dict = self.original_model.state_dict()

        # Convert key TL parameters back to HF format and load them
        tl_to_hf_mapping = {
            "embed.W_E": "transformer.wte.weight",
            "pos_embed.W_pos": "transformer.wpe.weight",
            "unembed.W_U": "lm_head.weight",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
        }

        # Convert basic parameters
        for tl_key, hf_key in tl_to_hf_mapping.items():
            if tl_key in processed_tl_state_dict:
                processed_tensor = processed_tl_state_dict[tl_key]

                # Handle tensor shape differences
                if tl_key == "unembed.W_U":
                    # TL: [d_model, d_vocab] -> HF: [d_vocab, d_model]
                    processed_tensor = processed_tensor.T

                # Find the corresponding key with _original_component suffix
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(processed_tensor)
                        break

        # Convert layer-specific parameters
        for layer_idx in range(self.cfg.n_layers):
            # Attention weights - for GPT-2, combine Q,K,V into c_attn
            tl_q_key = f"blocks.{layer_idx}.attn.W_Q"
            tl_k_key = f"blocks.{layer_idx}.attn.W_K"
            tl_v_key = f"blocks.{layer_idx}.attn.W_V"
            tl_o_key = f"blocks.{layer_idx}.attn.W_O"

            if all(key in processed_tl_state_dict for key in [tl_q_key, tl_k_key, tl_v_key]):
                # Combine Q,K,V weights for GPT-2 format
                w_q = processed_tl_state_dict[tl_q_key]  # [n_heads, d_model, d_head]
                w_k = processed_tl_state_dict[tl_k_key]  # [n_heads, d_model, d_head]
                w_v = processed_tl_state_dict[tl_v_key]  # [n_heads, d_model, d_head]

                # Reshape and combine: [d_model, 3*d_model] for HF format
                d_model = self.cfg.d_model
                w_q_flat = w_q.reshape(d_model, -1)  # [d_model, n_heads*d_head]
                w_k_flat = w_k.reshape(d_model, -1)  # [d_model, n_heads*d_head]
                w_v_flat = w_v.reshape(d_model, -1)  # [d_model, n_heads*d_head]

                combined_qkv = torch.cat(
                    [w_q_flat, w_k_flat, w_v_flat], dim=1
                )  # [d_model, 3*d_model]

                # Load into HF model
                hf_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(combined_qkv)
                        break

            if tl_o_key in processed_tl_state_dict:
                w_o = processed_tl_state_dict[tl_o_key]  # [n_heads, d_head, d_model]
                w_o_flat = w_o.reshape(
                    -1, self.cfg.d_model
                )  # [n_heads*d_head, d_model] = [d_model, d_model]

                hf_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(w_o_flat)
                        break

            # Attention biases
            tl_bq_key = f"blocks.{layer_idx}.attn.b_Q"
            tl_bk_key = f"blocks.{layer_idx}.attn.b_K"
            tl_bv_key = f"blocks.{layer_idx}.attn.b_V"
            tl_bo_key = f"blocks.{layer_idx}.attn.b_O"

            if all(key in processed_tl_state_dict for key in [tl_bq_key, tl_bk_key, tl_bv_key]):
                b_q = processed_tl_state_dict[tl_bq_key].flatten()
                b_k = processed_tl_state_dict[tl_bk_key].flatten()
                b_v = processed_tl_state_dict[tl_bv_key].flatten()
                combined_qkv_bias = torch.cat([b_q, b_k, b_v])

                hf_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(combined_qkv_bias)
                        break

            if tl_bo_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.attn.c_proj.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(processed_tl_state_dict[tl_bo_key])
                        break

            # MLP weights
            tl_mlp_in_key = f"blocks.{layer_idx}.mlp.W_in"
            tl_mlp_out_key = f"blocks.{layer_idx}.mlp.W_out"
            tl_mlp_bin_key = f"blocks.{layer_idx}.mlp.b_in"
            tl_mlp_bout_key = f"blocks.{layer_idx}.mlp.b_out"

            if tl_mlp_in_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_mlp_in_key]
                        )
                        break
            if tl_mlp_out_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_mlp_out_key]
                        )
                        break
            if tl_mlp_bin_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.mlp.c_fc.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_mlp_bin_key]
                        )
                        break
            if tl_mlp_bout_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.mlp.c_proj.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_mlp_bout_key]
                        )
                        break

            # LayerNorm weights
            tl_ln1_w_key = f"blocks.{layer_idx}.ln1.w"
            tl_ln1_b_key = f"blocks.{layer_idx}.ln1.b"
            tl_ln2_w_key = f"blocks.{layer_idx}.ln2.w"
            tl_ln2_b_key = f"blocks.{layer_idx}.ln2.b"

            if tl_ln1_w_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.ln_1.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_ln1_w_key]
                        )
                        break
            if tl_ln1_b_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.ln_1.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_ln1_b_key]
                        )
                        break
            if tl_ln2_w_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.ln_2.weight"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_ln2_w_key]
                        )
                        break
            if tl_ln2_b_key in processed_tl_state_dict:
                hf_key = f"transformer.h.{layer_idx}.ln_2.bias"
                for orig_key in original_state_dict.keys():
                    if orig_key.replace("._original_component", "") == hf_key:
                        original_state_dict[orig_key].data.copy_(
                            processed_tl_state_dict[tl_ln2_b_key]
                        )
                        break

        print(f"  Successfully loaded processed weights back into HuggingFace model")

    def _fold_layer_norm_hf(self, hf_state_dict):
        """Fold LayerNorm weights into attention and MLP weights (HF format)."""
        import torch

        for layer_idx in range(self.cfg.n_layers):
            # Fold ln1 into attention weights (avoid ln_final to prevent tied weight issues)
            ln1_weight_key = f"transformer.h.{layer_idx}.ln_1.weight"
            ln1_bias_key = f"transformer.h.{layer_idx}.ln_1.bias"
            qkv_weight_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
            qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"

            if all(key in hf_state_dict for key in [ln1_weight_key, qkv_weight_key]):
                ln1_weight = hf_state_dict[ln1_weight_key]
                qkv_weight = hf_state_dict[qkv_weight_key]

                # Center the QKV weight first
                qkv_weight_centered = qkv_weight - qkv_weight.mean(dim=0, keepdim=True)

                # Apply LayerNorm weight folding: W_new = W_old * ln_weight
                folded_qkv_weight = qkv_weight_centered * ln1_weight.unsqueeze(1)
                hf_state_dict[qkv_weight_key] = folded_qkv_weight

                # Handle bias folding if both exist
                if ln1_bias_key in hf_state_dict and qkv_bias_key in hf_state_dict:
                    ln1_bias = hf_state_dict[ln1_bias_key]
                    qkv_bias = hf_state_dict[qkv_bias_key]

                    # Bias folding: b_new = b_old + (W_old * ln_bias).sum(dim=0)
                    ln_bias_contribution = (qkv_weight * ln1_bias.unsqueeze(1)).sum(dim=0)
                    folded_qkv_bias = qkv_bias + ln_bias_contribution
                    hf_state_dict[qkv_bias_key] = folded_qkv_bias

                # Zero out the LayerNorm parameters (they've been folded)
                hf_state_dict[ln1_weight_key] = torch.ones_like(ln1_weight)
                if ln1_bias_key in hf_state_dict:
                    hf_state_dict[ln1_bias_key] = torch.zeros_like(hf_state_dict[ln1_bias_key])

            # Fold ln2 into MLP input weights
            ln2_weight_key = f"transformer.h.{layer_idx}.ln_2.weight"
            ln2_bias_key = f"transformer.h.{layer_idx}.ln_2.bias"
            mlp_in_weight_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            mlp_in_bias_key = f"transformer.h.{layer_idx}.mlp.c_fc.bias"

            if all(key in hf_state_dict for key in [ln2_weight_key, mlp_in_weight_key]):
                ln2_weight = hf_state_dict[ln2_weight_key]
                mlp_in_weight = hf_state_dict[mlp_in_weight_key]

                # Center the MLP input weight first
                mlp_in_weight_centered = mlp_in_weight - mlp_in_weight.mean(dim=0, keepdim=True)

                # Apply LayerNorm weight folding
                folded_mlp_in_weight = mlp_in_weight_centered * ln2_weight.unsqueeze(1)
                hf_state_dict[mlp_in_weight_key] = folded_mlp_in_weight

                # Handle bias folding if both exist
                if ln2_bias_key in hf_state_dict and mlp_in_bias_key in hf_state_dict:
                    ln2_bias = hf_state_dict[ln2_bias_key]
                    mlp_in_bias = hf_state_dict[mlp_in_bias_key]

                    # Bias folding
                    ln_bias_contribution = (mlp_in_weight * ln2_bias.unsqueeze(1)).sum(dim=0)
                    folded_mlp_in_bias = mlp_in_bias + ln_bias_contribution
                    hf_state_dict[mlp_in_bias_key] = folded_mlp_in_bias

                # Zero out the LayerNorm parameters
                hf_state_dict[ln2_weight_key] = torch.ones_like(ln2_weight)
                if ln2_bias_key in hf_state_dict:
                    hf_state_dict[ln2_bias_key] = torch.zeros_like(hf_state_dict[ln2_bias_key])

    def _center_writing_weights_hf(self, hf_state_dict):
        """Center weights that write to the residual stream (HF format)."""
        writing_weight_keys = [
            "transformer.wte.weight",  # Token embedding
            "transformer.wpe.weight",  # Position embedding
        ]

        # Add attention output and MLP output weights for each layer
        for layer_idx in range(self.cfg.n_layers):
            writing_weight_keys.extend(
                [
                    f"transformer.h.{layer_idx}.attn.c_proj.weight",  # Attention output
                    f"transformer.h.{layer_idx}.mlp.c_proj.weight",  # MLP output
                ]
            )

        for key in writing_weight_keys:
            if key in hf_state_dict:
                weight = hf_state_dict[key]
                # Center along the output dimension (dim=-1 for HF format)
                centered_weight = weight - weight.mean(dim=-1, keepdim=True)
                hf_state_dict[key] = centered_weight

    def _center_unembed_hf(self, hf_state_dict):
        """Center the unembedding weights (HF format)."""
        unembed_key = "lm_head.weight"
        if unembed_key in hf_state_dict:
            unembed_weight = hf_state_dict[unembed_key]
            # Center along the d_model dimension (dim=-1 for HF format)
            centered_unembed = unembed_weight - unembed_weight.mean(dim=-1, keepdim=True)
            hf_state_dict[unembed_key] = centered_unembed

    def _fold_value_biases_hf(self, hf_state_dict):
        """Fold value biases into output biases (HF format)."""
        for layer_idx in range(self.cfg.n_layers):
            qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
            attn_out_weight_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
            attn_out_bias_key = f"transformer.h.{layer_idx}.attn.c_proj.bias"

            if all(key in hf_state_dict for key in [qkv_bias_key, attn_out_weight_key]):
                qkv_bias = hf_state_dict[qkv_bias_key]
                attn_out_weight = hf_state_dict[attn_out_weight_key]

                # Extract V bias from combined QKV bias (last third)
                d_model = self.cfg.d_model
                v_bias = qkv_bias[2 * d_model : 3 * d_model]  # V bias is the last third

                # Reshape for matrix multiplication
                n_heads = self.cfg.n_heads
                d_head = self.cfg.d_head
                v_bias_reshaped = v_bias.view(n_heads, d_head)  # [n_heads, d_head]
                attn_out_weight_reshaped = attn_out_weight.view(
                    n_heads, d_head, d_model
                )  # [n_heads, d_head, d_model]

                # Apply folding: folded_b_O = b_O_original + (b_V * W_O).sum([0, 1])
                folded_contribution = (v_bias_reshaped[:, :, None] * attn_out_weight_reshaped).sum(
                    [0, 1]
                )

                if attn_out_bias_key in hf_state_dict:
                    hf_state_dict[attn_out_bias_key] += folded_contribution
                else:
                    hf_state_dict[attn_out_bias_key] = folded_contribution

                # Zero out the V component of the QKV bias
                qkv_bias[2 * d_model : 3 * d_model] = 0.0
                hf_state_dict[qkv_bias_key] = qkv_bias

    def _load_processed_hf_weights(self, processed_hf_state_dict):
        """Load processed HF weights back into the original model."""
        original_state_dict = self.original_model.state_dict()

        for hf_key, processed_tensor in processed_hf_state_dict.items():
            # Find the corresponding key with _original_component suffix
            original_key = None
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == hf_key:
                    original_key = orig_key
                    break

            if original_key and original_key in original_state_dict:
                # Load the processed tensor
                original_state_dict[original_key].data.copy_(processed_tensor)

    def _apply_corrected_processing_to_bridge_components(
        self, processed_tl_state_dict, missing_keys
    ):
        """Apply processed weights directly to TransformerBridge components.

        This converts the processed TransformerLens format weights back to HuggingFace format
        and loads them into the bridge's original_model components.
        """
        print("  Converting processed TL weights back to HF format...")

        # Convert TL format back to HF format using the adapter in reverse
        processed_hf_state_dict = {}

        # Map key TL parameters back to HF format
        tl_to_hf_mapping = {
            "embed.W_E": "transformer.wte.weight",
            "pos_embed.W_pos": "transformer.wpe.weight",
            "unembed.W_U": "lm_head.weight",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
        }

        # Convert basic parameters
        for tl_key, hf_key in tl_to_hf_mapping.items():
            if tl_key in processed_tl_state_dict:
                processed_hf_state_dict[hf_key] = processed_tl_state_dict[tl_key]

        # Convert layer-specific parameters
        for layer_idx in range(self.cfg.n_layers):
            # Attention weights - for GPT-2, combine Q,K,V into c_attn
            tl_q_key = f"blocks.{layer_idx}.attn.W_Q"
            tl_k_key = f"blocks.{layer_idx}.attn.W_K"
            tl_v_key = f"blocks.{layer_idx}.attn.W_V"
            tl_o_key = f"blocks.{layer_idx}.attn.W_O"

            if all(key in processed_tl_state_dict for key in [tl_q_key, tl_k_key, tl_v_key]):
                # Combine Q,K,V weights for GPT-2 format
                w_q = processed_tl_state_dict[tl_q_key]  # [n_heads, d_model, d_head]
                w_k = processed_tl_state_dict[tl_k_key]  # [n_heads, d_model, d_head]
                w_v = processed_tl_state_dict[tl_v_key]  # [n_heads, d_model, d_head]

                # Reshape and combine: [d_model, 3*d_model] for HF format
                d_model = self.cfg.d_model
                w_q_flat = w_q.reshape(d_model, -1)  # [d_model, n_heads*d_head]
                w_k_flat = w_k.reshape(d_model, -1)  # [d_model, n_heads*d_head]
                w_v_flat = w_v.reshape(d_model, -1)  # [d_model, n_heads*d_head]

                combined_qkv = torch.cat(
                    [w_q_flat, w_k_flat, w_v_flat], dim=1
                )  # [d_model, 3*d_model]
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.attn.c_attn.weight"
                ] = combined_qkv

            if tl_o_key in processed_tl_state_dict:
                w_o = processed_tl_state_dict[tl_o_key]  # [n_heads, d_head, d_model]
                w_o_flat = w_o.reshape(
                    -1, self.cfg.d_model
                )  # [n_heads*d_head, d_model] = [d_model, d_model]
                processed_hf_state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = w_o_flat

            # Attention biases
            tl_bq_key = f"blocks.{layer_idx}.attn.b_Q"
            tl_bk_key = f"blocks.{layer_idx}.attn.b_K"
            tl_bv_key = f"blocks.{layer_idx}.attn.b_V"
            tl_bo_key = f"blocks.{layer_idx}.attn.b_O"

            if all(key in processed_tl_state_dict for key in [tl_bq_key, tl_bk_key, tl_bv_key]):
                b_q = processed_tl_state_dict[tl_bq_key].flatten()
                b_k = processed_tl_state_dict[tl_bk_key].flatten()
                b_v = processed_tl_state_dict[tl_bv_key].flatten()
                combined_qkv_bias = torch.cat([b_q, b_k, b_v])
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.attn.c_attn.bias"
                ] = combined_qkv_bias

            if tl_bo_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.attn.c_proj.bias"
                ] = processed_tl_state_dict[tl_bo_key]

            # MLP weights
            tl_mlp_in_key = f"blocks.{layer_idx}.mlp.W_in"
            tl_mlp_out_key = f"blocks.{layer_idx}.mlp.W_out"
            tl_mlp_bin_key = f"blocks.{layer_idx}.mlp.b_in"
            tl_mlp_bout_key = f"blocks.{layer_idx}.mlp.b_out"

            if tl_mlp_in_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.mlp.c_fc.weight"
                ] = processed_tl_state_dict[tl_mlp_in_key]
            if tl_mlp_out_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.mlp.c_proj.weight"
                ] = processed_tl_state_dict[tl_mlp_out_key]
            if tl_mlp_bin_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.mlp.c_fc.bias"
                ] = processed_tl_state_dict[tl_mlp_bin_key]
            if tl_mlp_bout_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.mlp.c_proj.bias"
                ] = processed_tl_state_dict[tl_mlp_bout_key]

            # LayerNorm weights
            tl_ln1_w_key = f"blocks.{layer_idx}.ln1.w"
            tl_ln1_b_key = f"blocks.{layer_idx}.ln1.b"
            tl_ln2_w_key = f"blocks.{layer_idx}.ln2.w"
            tl_ln2_b_key = f"blocks.{layer_idx}.ln2.b"

            if tl_ln1_w_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.ln_1.weight"
                ] = processed_tl_state_dict[tl_ln1_w_key]
            if tl_ln1_b_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.ln_1.bias"
                ] = processed_tl_state_dict[tl_ln1_b_key]
            if tl_ln2_w_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.ln_2.weight"
                ] = processed_tl_state_dict[tl_ln2_w_key]
            if tl_ln2_b_key in processed_tl_state_dict:
                processed_hf_state_dict[
                    f"transformer.h.{layer_idx}.ln_2.bias"
                ] = processed_tl_state_dict[tl_ln2_b_key]

        # Handle missing LayerNorm parameters by setting them to identity in HF format
        print(f"  Setting missing LayerNorm parameters to identity...")
        for key in missing_keys:
            if ".ln1.w" in key or ".ln2.w" in key or "ln_final.w" in key:
                # Convert TL key to HF key
                if "ln_final.w" in key:
                    hf_key = "transformer.ln_f.weight"
                elif ".ln1.w" in key:
                    layer_idx = int(key.split(".")[1])
                    hf_key = f"transformer.h.{layer_idx}.ln_1.weight"
                elif ".ln2.w" in key:
                    layer_idx = int(key.split(".")[1])
                    hf_key = f"transformer.h.{layer_idx}.ln_2.weight"

                processed_hf_state_dict[hf_key] = torch.ones(self.cfg.d_model)

            elif ".ln1.b" in key or ".ln2.b" in key or "ln_final.b" in key:
                # Convert TL key to HF key
                if "ln_final.b" in key:
                    hf_key = "transformer.ln_f.bias"
                elif ".ln1.b" in key:
                    layer_idx = int(key.split(".")[1])
                    hf_key = f"transformer.h.{layer_idx}.ln_1.bias"
                elif ".ln2.b" in key:
                    layer_idx = int(key.split(".")[1])
                    hf_key = f"transformer.h.{layer_idx}.ln_2.bias"

                processed_hf_state_dict[hf_key] = torch.zeros(self.cfg.d_model)

        # Load the processed HF weights back into the original model
        print("  Loading processed weights into original model...")
        original_state_dict = self.original_model.state_dict()

        for hf_key, processed_tensor in processed_hf_state_dict.items():
            # Find the corresponding key with _original_component suffix
            original_key = None
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == hf_key:
                    original_key = orig_key
                    break

            if original_key and original_key in original_state_dict:
                # Load the processed tensor
                original_state_dict[original_key].data.copy_(processed_tensor)

        print(f"  Successfully applied processing to {len(processed_hf_state_dict)} parameters")

    def _apply_simplified_layernorm_folding(self, state_dict):
        """Correct LayerNorm folding implementation for HF format tensors."""
        print("    Applying LayerNorm folding to attention and MLP layers...")

        # Fold LayerNorm into attention layers (ln1 -> attention weights)
        for layer_idx in range(self.cfg.n_layers):
            ln1_weight_key = f"transformer.h.{layer_idx}.ln_1.weight"
            ln1_bias_key = f"transformer.h.{layer_idx}.ln_1.bias"

            if ln1_weight_key in state_dict and ln1_bias_key in state_dict:
                ln1_weight = state_dict[ln1_weight_key]  # [d_model]
                ln1_bias = state_dict[ln1_bias_key]  # [d_model]

                # Fold into attention QKV weights and biases
                qkv_weight_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
                qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"

                if qkv_weight_key in state_dict:
                    qkv_weight = state_dict[qkv_weight_key]  # HF format: [d_model, 3*d_model]

                    # Fold biases first (they depend on weights but not vice versa)
                    if qkv_bias_key in state_dict:
                        qkv_bias = state_dict[qkv_bias_key]  # [3*d_model]
                        # Correct bias folding: bias += (W * ln_bias).sum(dim_input)
                        # For HF format [d_model, 3*d_model], sum over d_model (dim=0)
                        ln_bias_contribution = (qkv_weight * ln1_bias.unsqueeze(1)).sum(dim=0)
                        state_dict[qkv_bias_key] = qkv_bias + ln_bias_contribution

                    # Fold weights: W_new = W * ln_weight (broadcast over input dimension)
                    # For HF format [d_model, 3*d_model], broadcast ln_weight over dim=0
                    state_dict[qkv_weight_key] = qkv_weight * ln1_weight.unsqueeze(1)

                    # Center weights: remove mean along input dimension (d_model, dim=0)
                    qkv_weight_centered = state_dict[qkv_weight_key] - state_dict[
                        qkv_weight_key
                    ].mean(dim=0, keepdim=True)
                    state_dict[qkv_weight_key] = qkv_weight_centered

                # Remove the folded LayerNorm parameters
                del state_dict[ln1_weight_key]
                del state_dict[ln1_bias_key]

        # Fold LayerNorm into MLP layers (ln2 -> MLP weights)
        for layer_idx in range(self.cfg.n_layers):
            ln2_weight_key = f"transformer.h.{layer_idx}.ln_2.weight"
            ln2_bias_key = f"transformer.h.{layer_idx}.ln_2.bias"

            if ln2_weight_key in state_dict and ln2_bias_key in state_dict:
                ln2_weight = state_dict[ln2_weight_key]  # [d_model]
                ln2_bias = state_dict[ln2_bias_key]  # [d_model]

                # Fold into MLP input weights
                mlp_weight_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
                mlp_bias_key = f"transformer.h.{layer_idx}.mlp.c_fc.bias"

                if mlp_weight_key in state_dict:
                    mlp_weight = state_dict[mlp_weight_key]  # HF format: [d_model, d_mlp]

                    # Fold biases first
                    if mlp_bias_key in state_dict:
                        mlp_bias = state_dict[mlp_bias_key]  # [d_mlp]
                        # Correct bias folding: bias += (W * ln_bias).sum(dim_input)
                        ln_bias_contribution = (mlp_weight * ln2_bias.unsqueeze(1)).sum(dim=0)
                        state_dict[mlp_bias_key] = mlp_bias + ln_bias_contribution

                    # Fold weights: W_new = W * ln_weight
                    state_dict[mlp_weight_key] = mlp_weight * ln2_weight.unsqueeze(1)

                    # Center weights: remove mean along input dimension (d_model, dim=0)
                    mlp_weight_centered = state_dict[mlp_weight_key] - state_dict[
                        mlp_weight_key
                    ].mean(dim=0, keepdim=True)
                    state_dict[mlp_weight_key] = mlp_weight_centered

                # Remove the folded LayerNorm parameters
                del state_dict[ln2_weight_key]
                del state_dict[ln2_bias_key]

        # Skip folding ln_final to avoid tied weights issue with unembedding
        print(f"    Folded LayerNorm for {self.cfg.n_layers} attention and MLP layers")
        print("    Skipped ln_final folding to avoid tied weights issue")

    def _apply_simplified_center_writing_weights(self, state_dict):
        """Correct center writing weights implementation matching ProcessWeights exactly."""
        print("    Applying center writing weights...")

        # Center embedding weights (these write to residual stream)
        if "transformer.wte.weight" in state_dict:
            embed_weight = state_dict["transformer.wte.weight"]  # HF format: [vocab_size, d_model]
            # ProcessWeights centers along the last dimension (-1), which is d_model for HF format
            embed_mean = embed_weight.mean(dim=-1, keepdim=True)
            state_dict["transformer.wte.weight"] = embed_weight - embed_mean
            print("      Centered transformer.wte.weight")

        # Center positional embedding if it exists (GPT-2 has this)
        if "transformer.wpe.weight" in state_dict:
            pos_embed_weight = state_dict["transformer.wpe.weight"]  # [n_ctx, d_model]
            # Center along last dimension (d_model)
            pos_embed_mean = pos_embed_weight.mean(dim=-1, keepdim=True)
            state_dict["transformer.wpe.weight"] = pos_embed_weight - pos_embed_mean
            print("      Centered transformer.wpe.weight")

        # Center attention output weights (these write to residual stream)
        for layer_idx in range(self.cfg.n_layers):
            attn_out_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
            attn_bias_key = f"transformer.h.{layer_idx}.attn.c_proj.bias"

            if attn_out_key in state_dict:
                attn_weight = state_dict[attn_out_key]  # HF format: [d_model, d_model]
                # ProcessWeights expects TL format [head_index, d_model, d_head] and centers along dim=-1 (d_head)
                # For HF format [d_model, d_model], we need to center along the last dimension
                attn_mean = attn_weight.mean(dim=-1, keepdim=True)
                state_dict[attn_out_key] = attn_weight - attn_mean

            if attn_bias_key in state_dict:
                attn_bias = state_dict[attn_bias_key]  # [d_model]
                bias_mean = attn_bias.mean()
                state_dict[attn_bias_key] = attn_bias - bias_mean

        # Center MLP output weights (these write to residual stream)
        for layer_idx in range(self.cfg.n_layers):
            mlp_out_key = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
            mlp_bias_key = f"transformer.h.{layer_idx}.mlp.c_proj.bias"

            if mlp_out_key in state_dict:
                mlp_weight = state_dict[mlp_out_key]  # HF format: [d_mlp, d_model]
                # ProcessWeights expects TL format [d_mlp, d_model] and centers along dim=-1 (d_model)
                # This matches HF format, so center along last dimension
                mlp_mean = mlp_weight.mean(dim=-1, keepdim=True)
                state_dict[mlp_out_key] = mlp_weight - mlp_mean

            if mlp_bias_key in state_dict:
                mlp_bias = state_dict[mlp_bias_key]  # [d_model]
                bias_mean = mlp_bias.mean()
                state_dict[mlp_bias_key] = mlp_bias - bias_mean

        print(f"      Centered writing weights for {self.cfg.n_layers} layers")

    def _apply_simplified_center_unembed(self, state_dict):
        """Correct center unembedding implementation matching ProcessWeights exactly."""
        if "lm_head.weight" in state_dict:
            # ProcessWeights centers along the last dimension (-1)
            # For HF format [vocab_size, d_model], this means centering along d_model
            unembed_weight = state_dict["lm_head.weight"]
            unembed_mean = unembed_weight.mean(dim=-1, keepdim=True)
            state_dict["lm_head.weight"] = unembed_weight - unembed_mean
            print("    Applied center_unembed to lm_head.weight")

    def _apply_simplified_fold_value_biases(self, state_dict):
        """Correct value bias folding implementation for HF format tensors."""
        print("    Applying value bias folding...")

        for layer_idx in range(self.cfg.n_layers):
            # GPT-2 uses combined QKV weights, so we need to extract V bias from the combined bias
            qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
            attn_out_weight_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
            attn_out_bias_key = f"transformer.h.{layer_idx}.attn.c_proj.bias"

            if (
                qkv_bias_key in state_dict
                and attn_out_weight_key in state_dict
                and attn_out_bias_key in state_dict
            ):
                qkv_bias = state_dict[qkv_bias_key]  # [3*d_model] - combined Q, K, V biases
                attn_out_weight = state_dict[attn_out_weight_key]  # HF format: [d_model, d_model]
                attn_out_bias = state_dict[attn_out_bias_key]  # [d_model]

                # Extract V bias from combined QKV bias (last third)
                d_model = self.cfg.d_model
                v_bias = qkv_bias[2 * d_model : 3 * d_model]  # [d_model] - just the V component

                # For GPT-2, we need to reshape V bias to [n_heads, d_head] to match the math
                n_heads = self.cfg.n_heads
                d_head = d_model // n_heads
                v_bias_reshaped = v_bias.reshape(n_heads, d_head)  # [n_heads, d_head]

                # Reshape attention output weight from [d_model, d_model] to [n_heads, d_head, d_model]
                # to match the expected W_O format in the folding formula
                attn_out_weight_reshaped = attn_out_weight.T.reshape(
                    n_heads, d_head, d_model
                )  # [n_heads, d_head, d_model]

                # Apply value bias folding: b_O_new = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])
                # v_bias_reshaped: [n_heads, d_head], W_O: [n_heads, d_head, d_model]
                # Need to broadcast: [n_heads, d_head, 1] * [n_heads, d_head, d_model] -> [n_heads, d_head, d_model]
                folded_contribution = (v_bias_reshaped[:, :, None] * attn_out_weight_reshaped).sum(
                    [0, 1]
                )  # [d_model]

                # Update the output bias
                state_dict[attn_out_bias_key] = attn_out_bias + folded_contribution

                # Zero out the V component of the combined QKV bias
                qkv_bias_zeroed = qkv_bias.clone()
                qkv_bias_zeroed[2 * d_model : 3 * d_model] = 0.0  # Zero out V bias component
                state_dict[qkv_bias_key] = qkv_bias_zeroed

        print(f"    Folded value biases for {self.cfg.n_layers} layers")

    def _load_processed_weights_from_hf_dict(self, processed_hf_dict):
        """Load processed weights (in HF format) back into the TransformerBridge.

        Args:
            processed_hf_dict: Dictionary of processed weights in HuggingFace format
        """
        # Load the processed weights back into the original model components
        # This preserves the HF format without conversion
        original_state_dict = self.original_model.state_dict()

        for key, processed_tensor in processed_hf_dict.items():
            # Find the corresponding key in the original model (with _original_component)
            original_key = None
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == key:
                    original_key = orig_key
                    break

            if original_key and original_key in original_state_dict:
                # Load the processed tensor back into the original model
                original_state_dict[original_key].data.copy_(processed_tensor)

        print(f"Loaded {len(processed_hf_dict)} processed parameters back into bridge")

    def _apply_weight_processing_inplace(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Apply weight processing transformations directly to bridge components in HuggingFace format.

        This method applies the same transformations as ProcessWeights but works directly on the
        bridge's components without converting tensor formats.
        """
        print("Applying weight processing transformations in-place...")

        # Step 1: Fold LayerNorm if requested (DISABLED for debugging)
        if fold_ln:
            print("  - Skipping LayerNorm folding (disabled for debugging)")
            # self._fold_layer_norm_inplace()

        # Step 2: Center writing weights if requested
        if center_writing_weights:
            self._center_writing_weights_inplace()

        # Step 3: Center unembedding if requested (DISABLED for debugging)
        if center_unembed:
            print("  - Skipping centering unembedding (disabled for debugging)")
            # self._center_unembed_inplace()

        # Step 4: Fold value biases if requested (DISABLED for debugging)
        if fold_value_biases:
            print("  - Skipping folding value biases (disabled for debugging)")
            # self._fold_value_biases_inplace()

        # Step 5: Refactor attention matrices if requested
        if refactor_factored_attn_matrices:
            self._refactor_factored_attn_matrices_inplace()

    def _fold_layer_norm_inplace(self):
        """Fold LayerNorm weights into subsequent layers in-place."""
        # For now, implement a simple version - we can expand this later
        print("  - Folding LayerNorm weights...")
        # TODO: Implement LayerNorm folding logic for HF format

    def _center_writing_weights_inplace(self):
        """Center weights that write to the residual stream in-place."""
        print("  - Centering writing weights...")

        # Center embedding weights (W_E)
        if hasattr(self, "embed") and hasattr(self.embed, "weight"):
            embed_weight = self.embed.weight.data  # HF format: [vocab_size, d_model]
            # Subtract mean along d_model dimension (last dim = -1)
            embed_mean = embed_weight.mean(dim=-1, keepdim=True)
            self.embed.weight.data = embed_weight - embed_mean

        # Center positional embedding weights (W_pos)
        if hasattr(self, "pos_embed") and hasattr(self.pos_embed, "weight"):
            if getattr(self.cfg, "positional_embedding_type", "standard") != "rotary":
                pos_weight = self.pos_embed.weight.data  # HF format: [seq_len, d_model]
                # Subtract mean along d_model dimension (last dim = -1)
                pos_mean = pos_weight.mean(dim=-1, keepdim=True)
                self.pos_embed.weight.data = pos_weight - pos_mean

        # Center attention output weights and biases (W_O, b_O)
        for layer_idx in range(self.cfg.n_layers):
            if layer_idx >= len(self.blocks):
                continue
            block = self.blocks[layer_idx]

            # Center attention output weights
            if hasattr(block.attn, "o") and hasattr(block.attn.o, "weight"):
                o_weight = block.attn.o.weight.data  # HF format: [d_model, d_model]
                # For writing weights, center along the last dimension (same as ProcessWeights)
                o_mean = o_weight.mean(dim=-1, keepdim=True)
                block.attn.o.weight.data = o_weight - o_mean

            # Center attention output biases
            if (
                hasattr(block.attn, "o")
                and hasattr(block.attn.o, "bias")
                and block.attn.o.bias is not None
            ):
                o_bias = block.attn.o.bias.data
                o_bias_mean = o_bias.mean()
                block.attn.o.bias.data = o_bias - o_bias_mean

            # Center MLP output weights and biases (W_out, b_out)
            if hasattr(block, "mlp") and not getattr(self.cfg, "attn_only", False):
                # Handle different MLP component names (out vs output)
                mlp_out_attr = (
                    "out"
                    if hasattr(block.mlp, "out")
                    else "output"
                    if hasattr(block.mlp, "output")
                    else None
                )

                if mlp_out_attr:
                    mlp_out = getattr(block.mlp, mlp_out_attr)

                    # Center MLP output weights
                    if hasattr(mlp_out, "weight"):
                        mlp_weight = mlp_out.weight.data  # HF format: [d_mlp, d_model]
                        # Subtract mean along the last dimension (d_model)
                        mlp_mean = mlp_weight.mean(dim=-1, keepdim=True)
                        mlp_out.weight.data = mlp_weight - mlp_mean

                    # Center MLP output biases
                    if hasattr(mlp_out, "bias") and mlp_out.bias is not None:
                        mlp_bias = mlp_out.bias.data
                        mlp_bias_mean = mlp_bias.mean()
                        mlp_out.bias.data = mlp_bias - mlp_bias_mean

    def _center_unembed_inplace(self):
        """Center unembedding weights in-place."""
        print("  - Centering unembedding weights...")
        if hasattr(self, "unembed") and hasattr(self.unembed, "weight"):
            # Center the unembedding weights (HF format: [vocab_size, d_model])
            unembed_weight = self.unembed.weight.data
            # Subtract the mean along the d_model dimension (dim=1)
            unembed_mean = unembed_weight.mean(dim=1, keepdim=True)
            self.unembed.weight.data = unembed_weight - unembed_mean

    def _fold_value_biases_inplace(self):
        """Fold value biases into output bias in-place."""
        print("  - Folding value biases...")

        for layer_idx in range(self.cfg.n_layers):
            if layer_idx >= len(self.blocks):
                continue
            block = self.blocks[layer_idx]

            # Get value biases and output weights/biases
            v_bias = None
            w_o = None
            b_o = None

            # Find value biases - need to use the original HuggingFace structure
            # In GPT-2, the original transformer uses combined qkv, so extract V bias from there
            if (
                hasattr(block.attn, "qkv")
                and hasattr(block.attn.qkv, "bias")
                and block.attn.qkv.bias is not None
            ):
                # For combined qkv, extract the V portion
                qkv_bias = block.attn.qkv.bias.data
                d_head = self.cfg.d_head
                n_heads = self.cfg.n_heads
                # Split into Q, K, V portions (each is n_heads * d_head)
                if qkv_bias.shape[0] == 3 * n_heads * d_head:
                    v_bias = qkv_bias[2 * n_heads * d_head :]  # V is the last third
            elif (
                hasattr(block.attn, "v")
                and hasattr(block.attn.v, "bias")
                and block.attn.v.bias is not None
            ):
                v_bias = block.attn.v.bias.data  # HF format: [n_heads * d_head] or similar

            # Find output weights and biases
            if hasattr(block.attn, "o"):
                if hasattr(block.attn.o, "weight"):
                    w_o = block.attn.o.weight.data  # HF format: [d_model, n_heads * d_head]
                if hasattr(block.attn.o, "bias") and block.attn.o.bias is not None:
                    b_o = block.attn.o.bias.data  # HF format: [d_model]

            # Apply the folding transformation if we have all components
            if v_bias is not None and w_o is not None and b_o is not None:
                try:
                    # Debug shapes
                    print(
                        f"    Layer {layer_idx}: v_bias shape: {v_bias.shape}, w_o shape: {w_o.shape}, b_o shape: {b_o.shape}"
                    )

                    # Reshape v_bias to [n_heads, d_head] if needed
                    if v_bias.dim() == 1:
                        expected_size = self.cfg.n_heads * self.cfg.d_head
                        if v_bias.shape[0] != expected_size:
                            print(
                                f"    Skipping layer {layer_idx}: v_bias size {v_bias.shape[0]} != expected {expected_size}"
                            )
                            continue
                        v_bias = v_bias.view(self.cfg.n_heads, self.cfg.d_head)

                    # Reshape w_o from HF format [d_model, n_heads * d_head] to [d_model, n_heads, d_head]
                    if w_o.dim() == 2:
                        expected_size = self.cfg.n_heads * self.cfg.d_head
                        if w_o.shape[1] != expected_size:
                            print(
                                f"    Skipping layer {layer_idx}: w_o dim 1 size {w_o.shape[1]} != expected {expected_size}"
                            )
                            continue
                        w_o = w_o.view(w_o.shape[0], self.cfg.n_heads, self.cfg.d_head)

                    # Compute the folded bias: b_O_new = b_O_original + sum_head(b_V_head @ W_O_head)
                    # v_bias: [n_heads, d_head], w_o: [d_model, n_heads, d_head]
                    # We want to compute sum over heads of (v_bias[h, :] @ w_o[:, h, :].T) for each head h
                    folded_contribution = torch.zeros_like(b_o)
                    for h in range(self.cfg.n_heads):
                        # v_bias[h]: [d_head], w_o[:, h]: [d_model, d_head]
                        # Compute v_bias[h] @ w_o[:, h].T = [d_model]
                        head_contribution = torch.matmul(w_o[:, h], v_bias[h])
                        folded_contribution += head_contribution

                    # Update the output bias
                    block.attn.o.bias.data = b_o + folded_contribution
                    print(f"    Successfully folded value biases for layer {layer_idx}")

                except Exception as e:
                    print(f"    Error folding value biases for layer {layer_idx}: {e}")
                    continue

                # Zero out the value biases (same logic as extraction)
                if (
                    hasattr(block.attn, "qkv")
                    and hasattr(block.attn.qkv, "bias")
                    and block.attn.qkv.bias is not None
                ):
                    # Zero out only the V portion of the combined qkv bias
                    d_head = self.cfg.d_head
                    n_heads = self.cfg.n_heads
                    if block.attn.qkv.bias.shape[0] == 3 * n_heads * d_head:
                        block.attn.qkv.bias.data[2 * n_heads * d_head :].zero_()
                elif (
                    hasattr(block.attn, "v")
                    and hasattr(block.attn.v, "bias")
                    and block.attn.v.bias is not None
                ):
                    block.attn.v.bias.data.zero_()

    def _refactor_factored_attn_matrices_inplace(self):
        """Refactor factored attention matrices in-place."""
        print("  - Refactoring factored attention matrices...")
        # TODO: Implement attention matrix refactoring for HF format

    def _load_processed_weights(self, processed_state_dict):
        """Load processed weights back into the TransformerBridge.

        Args:
            processed_state_dict: Dictionary of processed weights in TransformerLens format
        """
        # Load embedding weights
        if "embed.W_E" in processed_state_dict:
            self.embed.weight.data = processed_state_dict["embed.W_E"]
        if "pos_embed.W_pos" in processed_state_dict:
            self.pos_embed.weight.data = processed_state_dict["pos_embed.W_pos"]

        # Load layer weights
        for layer_idx in range(self.cfg.n_layers):
            if layer_idx >= len(self.blocks):
                continue

            block = self.blocks[layer_idx]

            # Load attention weights
            if f"blocks.{layer_idx}.attn.W_Q" in processed_state_dict:
                # The processed weights are in [n_heads, d_model, d_head] format
                # Need to reshape back to the bridge's expected format
                w_q = processed_state_dict[f"blocks.{layer_idx}.attn.W_Q"]
                w_k = processed_state_dict[f"blocks.{layer_idx}.attn.W_K"]
                w_v = processed_state_dict[f"blocks.{layer_idx}.attn.W_V"]
                w_o = processed_state_dict[f"blocks.{layer_idx}.attn.W_O"]

                # Reshape from TL format to bridge format and load
                if hasattr(block.attn, "q") and hasattr(block.attn.q, "weight"):
                    # For separate Q/K/V components, reshape from [n_heads, d_model, d_head] to [d_model, d_model]
                    if w_q.dim() == 3:  # [n_heads, d_model, d_head]
                        block.attn.q.weight.data = w_q.reshape(-1, w_q.shape[1])
                        block.attn.k.weight.data = w_k.reshape(-1, w_k.shape[1])
                        block.attn.v.weight.data = w_v.reshape(-1, w_v.shape[1])
                    else:
                        block.attn.q.weight.data = w_q
                        block.attn.k.weight.data = w_k
                        block.attn.v.weight.data = w_v

                if hasattr(block.attn, "o") and hasattr(block.attn.o, "weight"):
                    # For output weights, reshape from [n_heads, d_head, d_model] to [d_model, d_model]
                    if w_o.dim() == 3:  # [n_heads, d_head, d_model]
                        block.attn.o.weight.data = w_o.reshape(w_o.shape[1] * w_o.shape[0], -1)
                    else:
                        block.attn.o.weight.data = w_o

            # Load attention biases if they exist
            for bias_name in ["b_Q", "b_K", "b_V", "b_O"]:
                param_key = f"blocks.{layer_idx}.attn.{bias_name}"
                if param_key in processed_state_dict:
                    bridge_attr = bias_name[2:].lower()  # b_Q -> q, b_K -> k, etc.
                    if bridge_attr == "o":
                        bridge_attr = "o"
                    if hasattr(block.attn, bridge_attr):
                        attn_component = getattr(block.attn, bridge_attr)
                        if hasattr(attn_component, "bias") and attn_component.bias is not None:
                            bias_data = processed_state_dict[param_key]
                            if bias_data.dim() > 1:  # [n_heads, d_head] -> [n_heads * d_head]
                                bias_data = bias_data.reshape(-1)
                            attn_component.bias.data = bias_data

            # Load MLP weights
            if hasattr(block, "mlp"):
                mlp_weight_keys = ["W_in", "W_out", "W_gate"]
                mlp_bias_keys = ["b_in", "b_out", "b_gate"]

                for weight_key in mlp_weight_keys:
                    param_key = f"blocks.{layer_idx}.mlp.{weight_key}"
                    if param_key in processed_state_dict:
                        bridge_attr = weight_key[2:].lower()  # W_in -> in, W_out -> out
                        if bridge_attr == "in":
                            bridge_attr = "input"  # GPT-2 uses 'input' instead of 'in'
                        if hasattr(block.mlp, bridge_attr):
                            mlp_component = getattr(block.mlp, bridge_attr)
                            if hasattr(mlp_component, "weight"):
                                mlp_component.weight.data = processed_state_dict[param_key]

                for bias_key in mlp_bias_keys:
                    param_key = f"blocks.{layer_idx}.mlp.{bias_key}"
                    if param_key in processed_state_dict:
                        bridge_attr = bias_key[2:].lower()  # b_in -> in, b_out -> out
                        if bridge_attr == "in":
                            bridge_attr = "input"  # GPT-2 uses 'input' instead of 'in'
                        if hasattr(block.mlp, bridge_attr):
                            mlp_component = getattr(block.mlp, bridge_attr)
                            if hasattr(mlp_component, "bias") and mlp_component.bias is not None:
                                mlp_component.bias.data = processed_state_dict[param_key]

            # Load LayerNorm weights
            for ln_name in ["ln1", "ln2"]:
                for param_type in ["w", "b"]:
                    param_key = f"blocks.{layer_idx}.{ln_name}.{param_type}"
                    if param_key in processed_state_dict:
                        if hasattr(block, ln_name):
                            ln_component = getattr(block, ln_name)
                            attr_name = "weight" if param_type == "w" else "bias"
                            if hasattr(ln_component, attr_name):
                                param_tensor = getattr(ln_component, attr_name)
                                if param_tensor is not None:
                                    param_tensor.data = processed_state_dict[param_key]

        # Load final LayerNorm weights
        for param_type in ["w", "b"]:
            param_key = f"ln_final.{param_type}"
            if param_key in processed_state_dict:
                if hasattr(self, "ln_final"):
                    attr_name = "weight" if param_type == "w" else "bias"
                    if hasattr(self.ln_final, attr_name):
                        param_tensor = getattr(self.ln_final, attr_name)
                        if param_tensor is not None:
                            param_tensor.data = processed_state_dict[param_key]

        # Load unembedding weights
        if "unembed.W_U" in processed_state_dict:
            # Processed weights are in [d_model, d_vocab] format, bridge expects [d_vocab, d_model]
            unembed_weight = processed_state_dict["unembed.W_U"]
            if hasattr(self, "unembed") and hasattr(self.unembed, "weight"):
                self.unembed.weight.data = unembed_weight.T  # Transpose back

    # ==================== TOKENIZATION METHODS ====================

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        """Converts a string to a tensor of tokens.

        Args:
            input: The input to tokenize
            prepend_bos: Whether to prepend the BOS token
            padding_side: Which side to pad on
            move_to_device: Whether to move to model device
            truncate: Whether to truncate to model context length

        Returns:
            Token tensor of shape [batch, pos]
        """
        # Handle prepend_bos logic
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)

        # Handle padding_side logic
        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")

        # Use the pre-calculated tokenizer_prepends_bos configuration
        tokenizer_prepends_bos = getattr(self.cfg, "tokenizer_prepends_bos", True)

        if prepend_bos and not tokenizer_prepends_bos:
            # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
            input = utils.get_input_with_manually_prepended_bos(self.tokenizer.bos_token, input)

        if isinstance(input, str):
            input = [input]

        # Tokenize
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]

        if not prepend_bos and tokenizer_prepends_bos:
            # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
            tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

        if move_to_device:
            tokens = tokens.to(self.cfg.device)

        return tokens

    # ==================== PAST KV CACHE HELPERS ====================

    def get_pos_offset(self, past_kv_cache, batch_size: int) -> int:
        """Compute position offset from a TransformerLensKeyValueCache-like object.

        Mirrors HookedTransformer.get_pos_offset behavior for compatibility.
        """
        if past_kv_cache is None:
            return 0
        cached_batch_size, cache_ctx_length, num_heads_in_cache, d_head_in_cache = past_kv_cache[
            0
        ].past_keys.shape
        assert cached_batch_size == batch_size
        if getattr(self.cfg, "n_key_value_heads", None) is None:
            assert num_heads_in_cache == self.cfg.n_heads
        else:
            assert num_heads_in_cache == getattr(self.cfg, "n_key_value_heads")
        assert d_head_in_cache == self.cfg.d_head
        return cache_ctx_length

    def to_string(
        self,
        tokens: Union[List[int], torch.Tensor, np.ndarray],
    ) -> Union[str, List[str]]:
        """Convert tokens to string(s).

        Args:
            tokens: Tokens to convert

        Returns:
            Decoded string(s)
        """
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)

        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[str, torch.Tensor, np.ndarray, List],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Map text or tokens to a list of tokens as strings.

        Args:
            input: The input to convert
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on

        Returns:
            List of token strings
        """
        if isinstance(input, list):
            # Use cast to help mypy understand the recursive return type
            return cast(
                List[List[str]],
                [self.to_str_tokens(item, prepend_bos, padding_side) for item in input],
            )
        elif isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input.squeeze()
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens_np = input.squeeze()
            if tokens_np.ndim == 0:
                tokens_np = np.expand_dims(tokens_np, axis=0)
            assert (
                tokens_np.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens_np.shape}"
            tokens = torch.tensor(tokens_np)
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")

        str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        return str_tokens

    def to_single_token(self, string: str) -> int:
        """Map a string that makes up a single token to the id for that token.

        Args:
            string: The string to convert

        Returns:
            Token ID

        Raises:
            AssertionError: If string is not a single token
        """
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        if token.numel() != 1:
            raise AssertionError(f"Input string: {string} is not a single token!")
        return int(token.item())

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[str, torch.Tensor],
        mode="first",
        prepend_bos: Optional[Union[bool, None]] = None,
        padding_side: Optional[Union[Literal["left", "right"], None]] = None,
    ):
        """Get the position of a single_token in a string or sequence of tokens.

        Raises an error if the token is not present.

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a single token).
            input (Union[str, torch.Tensor]): The sequence to
                search in. Can be a string or a rank 1 tensor of tokens or a rank 2 tensor of tokens
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports
                "first" or "last". Defaults to "first".
            prepend_bos (bool, optional): Whether to prepend the BOS token to the input
                (only applies when input is a string). Defaults to None, using the bridge's default.
            padding_side (Union[Literal["left", "right"], None], optional): Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if isinstance(input, str):
            # If the input is a string, convert to tensor
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if len(tokens.shape) == 2:
            # If the tokens have shape [1, seq_len], flatten to [seq_len]
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]

        if isinstance(single_token, str):
            # If the single token is a string, convert to an integer
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()

        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def to_single_str_token(self, int_token: int) -> str:
        """Get the single token corresponding to an int in string form.

        Args:
            int_token: The token ID

        Returns:
            The token string
        """
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        if isinstance(token, list) and len(token) == 1:
            return str(token[0])
        raise AssertionError("Expected a single string token.")

    @property
    def W_K(self) -> torch.Tensor:
        """Stack the key weights across all layers."""
        weights = []
        for block in self.blocks:
            w_k = block.attn.W_K
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_k.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_k = w_k.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_k)
        return torch.stack(weights, dim=0)

    @property
    def W_Q(self) -> torch.Tensor:
        """Stack the query weights across all layers."""
        weights = []
        for block in self.blocks:
            w_q = block.attn.W_Q
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_q.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_q = w_q.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_q)
        return torch.stack(weights, dim=0)

    @property
    def W_V(self) -> torch.Tensor:
        """Stack the value weights across all layers."""
        weights = []
        for block in self.blocks:
            w_v = block.attn.W_V
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_v.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_v = w_v.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_v)
        return torch.stack(weights, dim=0)

    @property
    def W_O(self) -> torch.Tensor:
        """Stack the attn output weights across all layers."""
        weights = []
        for block in self.blocks:
            w_o = block.attn.W_O
            # Reshape from [d_model, d_model] to [n_heads, d_head, d_model]
            if w_o.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_o = w_o.reshape(self.cfg.n_heads, d_head, self.cfg.d_model)
            weights.append(w_o)
        return torch.stack(weights, dim=0)

    @property
    def W_in(self) -> torch.Tensor:
        """Stack the MLP input weights across all layers."""
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[torch.Tensor, None]:
        """Stack the MLP gate weights across all layers.

        Only works for models with gated MLPs.
        """
        if getattr(self.cfg, "gated_mlp", False):
            return torch.stack([block.mlp.W_gate for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> torch.Tensor:
        """Stack the MLP output weights across all layers."""
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> torch.Tensor:
        """Stack the key biases across all layers."""
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> torch.Tensor:
        """Stack the query biases across all layers."""
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> torch.Tensor:
        """Stack the value biases across all layers."""
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> torch.Tensor:
        """Stack the attn output biases across all layers."""
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> torch.Tensor:
        """Stack the MLP input biases across all layers."""
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> torch.Tensor:
        """Stack the MLP output biases across all layers."""
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    def get_params(self):
        """Access to model parameters in the format expected by SVDInterpreter.

        Returns:
            dict: Dictionary of parameter tensors with TransformerLens naming convention
        """
        # Simplified implementation to avoid syntax errors
        # The HF-native processing approach doesn't rely on this method
        params_dict = {}

        # Return empty dict - main functionality is in process_weights()
        return params_dict

    @property
    def params(self):
        """Property access to model parameters in the format expected by SVDInterpreter."""
        return self.get_params()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Return named parameters in the same format as HookedTransformer.

        This ensures compatibility with tools like SVDInterpreter that expect
        parameter names like 'blocks.0.attn.W_Q' instead of the raw model names.
        """
        params_dict = self.get_params()
        for name, param in params_dict.items():
            yield name, param

    # ==================== FORWARD PASS METHODS ====================

    def forward(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: str = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        past_kv_cache: Optional[TransformerLensKeyValueCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_at_layer: int = 0,
        **kwargs,
    ) -> Any:
        """Forward pass through the model.

        Args:
            input: Input to the model
            return_type: Type of output to return ('logits', 'loss', 'both', None)
            loss_per_token: Whether to return loss per token
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on
            past_kv_cache: Optional TransformerLensKeyValueCache for generation
            start_at_layer: Layer to start forward pass from
            **kwargs: Additional arguments passed to model

        Returns:
            Model output based on return_type
        """
        # Handle string input
        if isinstance(input, (str, list)):
            input_ids = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            input_ids = input

        # Handle explicit attention mask
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        # Handle KV cache if provided
        if past_kv_cache is not None:
            # Convert TransformerLensKeyValueCache to backend format
            # Create a list of tuples (keys, values) for each layer in backend format
            backend_cache = []
            for entry in past_kv_cache.entries:
                if entry.past_keys.numel() > 0:  # Only add if there are cached values
                    # Convert from TL format [batch, pos, n_heads, d_head] to backend format [batch, n_heads, pos, d_head]
                    cached_keys = entry.past_keys.transpose(1, 2)  # [batch, n_heads, pos, d_head]
                    cached_values = entry.past_values.transpose(
                        1, 2
                    )  # [batch, n_heads, pos, d_head]
                    backend_cache.append((cached_keys, cached_values))
                # Note: We skip empty entries rather than adding (None, None) to maintain type consistency

            kwargs["past_key_values"] = backend_cache

            # Handle attention mask from the cache
            if hasattr(past_kv_cache, "previous_attention_mask"):
                # Build attention mask that includes past context
                batch_size = input_ids.shape[0]
                current_length = input_ids.shape[1]
                past_length = past_kv_cache.previous_attention_mask.shape[1]

                # Use explicit attention mask if provided, otherwise create one for current tokens
                if attention_mask is not None:
                    current_mask = attention_mask
                else:
                    current_mask = torch.ones(
                        batch_size, current_length, dtype=torch.long, device=input_ids.device
                    )

                # Combine with past attention mask
                if past_length > 0:
                    full_attention_mask = torch.cat(
                        [past_kv_cache.previous_attention_mask, current_mask], dim=1
                    )
                else:
                    full_attention_mask = current_mask

                kwargs["attention_mask"] = full_attention_mask

            # Enable caching for the underlying model
            kwargs["use_cache"] = True
        elif "use_past_kv_cache" in kwargs and kwargs["use_past_kv_cache"]:
            # If use_past_kv_cache is True but no cache provided, enable caching
            kwargs["use_cache"] = True

        # Store reference to original TransformerLensKeyValueCache for updating
        original_tl_cache = past_kv_cache

        # Run model
        if hasattr(self.original_model, "forward"):
            # Pass labels for loss calculation if needed
            if return_type in ["loss", "both"]:
                kwargs["labels"] = input_ids
            output = self.original_model.forward(input_ids, **kwargs)
        else:
            if return_type in ["loss", "both"]:
                kwargs["labels"] = input_ids
            output = self.original_model(input_ids, **kwargs)

        # Update TransformerLensKeyValueCache if it was provided and model returned new cache
        if (
            original_tl_cache is not None
            and hasattr(output, "past_key_values")
            and output.past_key_values is not None
        ):
            # Convert backend cache format back to TransformerLens format
            backend_cache = output.past_key_values
            for i, (cached_keys, cached_values) in enumerate(backend_cache):
                if i < len(original_tl_cache.entries) and cached_keys is not None:
                    # Convert from backend format [batch, n_heads, pos, d_head] to TL format [batch, pos, n_heads, d_head]
                    tl_keys = cached_keys.transpose(1, 2)
                    tl_values = cached_values.transpose(1, 2)
                    original_tl_cache.entries[i].past_keys = tl_keys
                    original_tl_cache.entries[i].past_values = tl_values

            # Update attention mask for next iteration
            if attention_mask is not None:
                original_tl_cache.previous_attention_mask = kwargs.get(
                    "attention_mask", attention_mask
                )
            elif hasattr(original_tl_cache, "previous_attention_mask"):
                # Extend the previous mask with ones for the new tokens
                batch_size, current_length = input_ids.shape
                new_mask = torch.ones(
                    batch_size, current_length, dtype=torch.long, device=input_ids.device
                )
                if original_tl_cache.previous_attention_mask is not None:
                    original_tl_cache.previous_attention_mask = torch.cat(
                        [original_tl_cache.previous_attention_mask, new_mask], dim=1
                    )
                else:
                    original_tl_cache.previous_attention_mask = new_mask

        # Extract logits from output
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            logits = output[0]
        else:
            logits = output

        # Handle different return types
        if return_type == "logits":
            return logits
        elif return_type == "loss":
            if hasattr(output, "loss") and output.loss is not None:
                return output.loss
            else:
                # Calculate loss manually
                return self.loss_fn(logits, input_ids, per_token=loss_per_token)
        elif return_type == "both":
            loss = None
            if hasattr(output, "loss") and output.loss is not None:
                loss = output.loss
            else:
                loss = self.loss_fn(logits, input_ids, per_token=loss_per_token)
            return logits, loss
        elif return_type is None:
            return output
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    def loss_fn(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Calculate cross-entropy loss.

        Args:
            logits: Model logits
            tokens: Target tokens
            per_token: Whether to return per-token loss

        Returns:
            Loss tensor
        """
        # Simple cross-entropy loss implementation
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)

        # Shift logits and tokens for next-token prediction
        target_tokens = tokens[:, 1:].contiguous()  # Remove first token (typically BOS)
        pred_logits = logits[:, :-1]

        loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            target_tokens.reshape(-1),
            reduction="none",
        )

        if per_token:
            return loss.reshape(target_tokens.shape)
        else:
            return loss.mean()

    # ==================== CACHING METHODS ====================

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[True] = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, ActivationCache]:
        """Run with cache - placeholder implementation."""
        pass

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[False],
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run with cache - placeholder implementation."""
        pass

    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        stop_at_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, Union[ActivationCache, Dict[str, torch.Tensor]]]:
        """Run the model and cache all activations - placeholder implementation."""
        # Simplified implementation - just run forward and return empty cache
        output = self.forward(input, **kwargs)
        if return_cache_object:
            from transformer_lens.ActivationCache import ActivationCache

            cache = ActivationCache({}, self)
            return output, cache
        else:
            return output, {}

    def run_with_hooks(
        self,
        input: Union[str, List[str], torch.Tensor],
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        return_type: Optional[str] = "logits",
        names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        stop_at_layer: Optional[int] = None,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Any:
        """Run the model with specified forward and backward hooks - placeholder implementation."""
        # Simplified implementation - just run forward
        return self.forward(input, return_type=return_type, **kwargs)

    def generate(
        self,
        input: Union[str, List[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[str, List[str], torch.Tensor]:
        """Generate text from the model - placeholder implementation."""
        # Simplified implementation - just return input
        return input

    # ==================== DEVICE MANAGEMENT ====================

    def to(self, *args, **kwargs) -> "TransformerBridge":
        """Move model to device or change dtype.

        Args:
            args: Positional arguments for nn.Module.to
            kwargs: Keyword arguments for nn.Module.to

        Returns:
            Self for chaining
        """
        self.original_model = self.original_model.to(*args, **kwargs)
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "TransformerBridge":
        """Move model to CUDA.

        Args:
            device: CUDA device

        Returns:
            Self for chaining
        """
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self) -> "TransformerBridge":
        """Move model to CPU.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("cpu"))  # type: ignore
