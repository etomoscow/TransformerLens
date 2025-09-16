"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from typing import Any, cast

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import HookConversionSet
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.types import (
    ComponentMapping,
    RemoteComponent,
    RemoteModel,
    RemotePath,
    TransformerLensPath,
)


class ArchitectureAdapter:
    """Base class for architecture adapters.

    This class provides the interface for adapting between different model architectures.
    It handles both component mapping (for accessing model parts) and weight conversion
    (for initializing weights from one format to another).
    """

    default_cfg: dict[str, Any] = {}

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        """Initialize the architecture adapter.

        Args:
            cfg: The configuration object.
        """
        self.cfg = cfg

        self.component_mapping: ComponentMapping | None = None
        self.conversion_rules: HookConversionSet | None = None

        # Configuration for attention weight handling
        self.uses_split_attention: bool = getattr(cfg, "uses_split_attention", False)

        # Merge default_cfg into cfg for missing variables
        self._merge_default_config()

    def _merge_default_config(self) -> None:
        """Merge default_cfg into cfg for variables that don't exist in cfg."""
        for key, value in self.default_cfg.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

    def get_component_mapping(self) -> ComponentMapping:
        """Get the full component mapping.

        Returns:
            The component mapping dictionary

        Raises:
            ValueError: If the component mapping is not set
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component_mapping")
        return self.component_mapping

    def get_remote_component(self, model: RemoteModel, path: RemotePath) -> RemoteComponent:
        """Get a component from a remote model by its path.

        This method should be overridden by subclasses to provide the logic for
        accessing components in a specific model architecture.

        Args:
            model: The remote model
            path: The path to the component in the remote model's format

        Returns:
            The component (e.g., a PyTorch module)

        Raises:
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed
            ValueError: If the path is empty or invalid

        Examples:
            Get an embedding component:

            >>> # adapter.get_remote_component(model, "model.embed_tokens")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_remote_component(model, "model.layers.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_remote_component(model, "model.layers.0.ln1")
            >>> # <LayerNorm>
        """

        current = model
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]  # type: ignore[index]
            else:
                current = getattr(current, part)
        return current

    def get_component_from_list_module(
        self,
        list_module: RemoteComponent,
        bridge_component: GeneralizedComponent,
        parts: list[str],
    ) -> RemoteComponent:
        """Get a component from a list module using the bridge component and the transformer lens path.
        Args:
            list_module: The remote list module to get the component from
            bridge_component: The bridge component
            parts: The parts of the transformer lens path to navigate
        Returns:
            The requested component from the list module described by the path
        """

        # Handle list item indexing (like blocks)
        item_index = parts[1]
        if not item_index.isdigit():
            raise ValueError(f"Expected item index, got {item_index}")

        if not hasattr(list_module, "__getitem__"):
            raise TypeError(f"Component {bridge_component.name} is not indexable")

        # Cast to indicate to mypy that list_module is indexable after the check
        indexable_container = cast(Any, list_module)
        item = indexable_container[int(item_index)]

        if len(parts) == 2:
            # Just return the item
            return item
        else:
            # Get subcomponent from the item using bridge mapping
            subcomponent_name = parts[2]

            # Check the submodules attribute for bridge submodules
            if subcomponent_name in bridge_component.submodules:
                subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                # If there are more parts (like blocks.0.attn.W_Q), navigate deeper
                if len(parts) > 3:
                    # Navigate through the deeper subcomponents
                    current_bridge = subcomponent_bridge
                    current = getattr(item, subcomponent_bridge.name)

                    for i in range(3, len(parts)):
                        deeper_component_name = parts[i]

                        if deeper_component_name.isdigit() and current_bridge.is_list_item:
                            # We are dealing with a nested BlockBridge, call the function recursively
                            # and pass the path (parts) starting from the nested BlockBridge
                            return self.get_component_from_list_module(
                                current, current_bridge, parts[i - 1 :]
                            )

                        # Check submodules for deeper components
                        if deeper_component_name in current_bridge.submodules:
                            current_bridge = current_bridge.submodules[deeper_component_name]
                            current = getattr(current, current_bridge.name)
                        else:
                            raise ValueError(
                                f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                            )

                    return current
                else:
                    # Just the 3-level path
                    return getattr(item, subcomponent_bridge.name)
            else:
                raise ValueError(
                    f"Component {subcomponent_name} not found in {parts[0]} components"
                )

    def get_component(self, model: RemoteModel, path: TransformerLensPath) -> RemoteComponent:
        """Get a component from the model using the component_mapping.

        Args:
            model: The model to extract components from
            path: The path of the component to get, as defined in component_mapping

        Returns:
            The requested component from the model

        Raises:
            ValueError: If component_mapping is not set or if the component is not found
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed

        Examples:
            Get an embedding component:

            >>> # adapter.get_component(model, "embed")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_component(model, "blocks.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_component(model, "blocks.0.ln1")
            >>> # <LayerNorm>
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")

        # In the new system, we get the bridge component from the mapping
        # and use its name attribute to get the remote component
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if self.component_mapping is None or parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the component at the bridge's remote path
            return self.get_remote_component(model, bridge_component.name)

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if bridge_component.is_list_item and len(parts) >= 2:
            # Get the remote ModuleList for the indexed item
            list_module = self.get_remote_component(model, bridge_component.name)
            return self.get_component_from_list_module(list_module, bridge_component, parts)

        # For other nested paths, navigate through the remote model
        remote_path = bridge_component.name
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        return self.get_remote_component(model, remote_path)

    def translate_transformer_lens_path(
        self, path: TransformerLensPath, last_component_only: bool = False
    ) -> RemotePath:
        """Translate a TransformerLens path to a remote model path.

        Args:
            path: The TransformerLens path to translate
            last_component_only: If True, return only the last component of the path

        Returns:
            The corresponding remote model path

        Raises:
            ValueError: If the path is not found in the component mapping
        """
        if self.component_mapping is None:
            raise ValueError(
                "component_mapping must be set before calling translate_transformer_lens_path"
            )

        # Preprocess the path to handle parameter name mapping
        path, param_suffix = self._preprocess_parameter_path(path)

        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the bridge's remote path
            remote_path = bridge_component.name
            # Add parameter suffix from preprocessing
            if param_suffix:
                remote_path = remote_path + param_suffix
            if last_component_only:
                return remote_path.split(".")[-1]
            return remote_path

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if bridge_component.is_list_item and len(parts) >= 2:
            # Handle list item indexing (like blocks)
            item_index = parts[1]
            if not item_index.isdigit():
                raise ValueError(f"Expected item index, got {item_index}")

            # Get the base items path
            items_path = bridge_component.name

            if len(parts) == 2:
                # Just return the indexed item path
                remote_path = f"{items_path}.{item_index}"
                # Add parameter suffix from preprocessing
                if param_suffix:
                    remote_path = remote_path + param_suffix
                if last_component_only:
                    return remote_path.split(".")[-1]
                return remote_path
            else:
                # Get subcomponent from the item bridge
                subcomponent_name = parts[2]

                # Check the submodules attribute for bridge submodules
                if subcomponent_name in bridge_component.submodules:
                    subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                    # If there are more parts (like blocks.0.attn.q_proj), navigate deeper
                    if len(parts) > 3:
                        # Navigate through the deeper subcomponents
                        current_bridge = subcomponent_bridge
                        remote_path_parts = [items_path, item_index, subcomponent_bridge.name]

                        for i in range(3, len(parts)):
                            deeper_component_name = parts[i]

                            # Check submodules for deeper components
                            if deeper_component_name in current_bridge.submodules:
                                current_bridge = current_bridge.submodules[deeper_component_name]
                                remote_path_parts.append(current_bridge.name)
                            else:
                                raise ValueError(
                                    f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                                )

                        remote_path = ".".join(remote_path_parts)
                        # Add parameter suffix from preprocessing
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                    else:
                        # Just the 3-level path
                        remote_path = f"{items_path}.{item_index}.{subcomponent_bridge.name}"
                        # Add parameter suffix from preprocessing
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                else:
                    raise ValueError(
                        f"Component {subcomponent_name} not found in {parts[0]} components"
                    )

        # For other nested paths, navigate through the bridge components
        remote_path = bridge_component.name
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        # Add parameter suffix from preprocessing
        if param_suffix:
            remote_path = remote_path + param_suffix

        if last_component_only:
            return remote_path.split(".")[-1]
        return remote_path

    def _preprocess_parameter_path(self, path: str) -> tuple[str, str]:
        """Preprocess TransformerLens path to map parameter names to component names.

        Args:
            path: The original TransformerLens path

        Returns:
            Tuple of (preprocessed_path, parameter_suffix)
        """
        # Determine parameter suffix from the original path
        param_suffix = ""  # Initialize to handle all code paths
        if path.endswith(
            (
                ".W_Q",
                ".W_K",
                ".W_V",
                ".W_O",
                ".W_in",
                ".W_out",
                ".W_gate",
                ".W_E",
                ".W_U",
                ".W_pos",
                ".w",
            )
        ):
            param_suffix = ".weight"
        elif path.endswith(
            (
                ".b_Q",
                ".b_K",
                ".b_V",
                ".b_O",
                ".b_in",
                ".b_out",
                ".b_gate",
                ".b_E",
                ".b_U",
                ".b_pos",
                ".b",
            )
        ):
            param_suffix = ".bias"

        # Handle attention weights based on actual architecture
        # Check if this is an attention weight that needs architecture-specific mapping
        if any(
            path.endswith(suffix) for suffix in [".W_Q", ".W_K", ".W_V", ".b_Q", ".b_K", ".b_V"]
        ):
            # Extract the attention component path (e.g., "blocks.0.attn")
            attn_path_parts = path.split(".")
            if len(attn_path_parts) >= 3 and attn_path_parts[-2] == "attn":
                attn_component_path = ".".join(attn_path_parts[:-1])  # e.g., "blocks.0.attn"

                # Check what attention components are actually available
                try:
                    if self.component_mapping:
                        # Navigate to the attention component to see what submodules it has
                        current_mapping = self.component_mapping
                        for part in attn_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]

                        # Check available attention subcomponents
                        if hasattr(current_mapping, "submodules"):
                            attn_components = list(current_mapping.submodules.keys())

                            # If we have a combined qkv component, map all Q/K/V to it
                            if "qkv" in attn_components:
                                path = path.replace(".W_Q", ".qkv")
                                path = path.replace(".W_K", ".qkv")
                                path = path.replace(".W_V", ".qkv")
                                path = path.replace(".b_Q", ".qkv")
                                path = path.replace(".b_K", ".qkv")
                                path = path.replace(".b_V", ".qkv")
                            # If we have separate q, k, v components, map individually
                            elif all(comp in attn_components for comp in ["q", "k", "v"]):
                                path = path.replace(".W_Q", ".q")
                                path = path.replace(".W_K", ".k")
                                path = path.replace(".W_V", ".v")
                                path = path.replace(".b_Q", ".q")
                                path = path.replace(".b_K", ".k")
                                path = path.replace(".b_V", ".v")
                            # If we have qkv_proj (like some other architectures), use that
                            elif "qkv_proj" in attn_components:
                                path = path.replace(".W_Q", ".qkv_proj")
                                path = path.replace(".W_K", ".qkv_proj")
                                path = path.replace(".W_V", ".qkv_proj")
                                path = path.replace(".b_Q", ".qkv_proj")
                                path = path.replace(".b_K", ".qkv_proj")
                                path = path.replace(".b_V", ".qkv_proj")
                except Exception:
                    # Fallback to default behavior if component mapping inspection fails
                    pass

        # If no architecture-specific mapping was applied, use default fallback
        if any(
            path.endswith(suffix) for suffix in [".W_Q", ".W_K", ".W_V", ".b_Q", ".b_K", ".b_V"]
        ):
            # Default fallback - assume separate components
            path = path.replace(".W_Q", ".q")
            path = path.replace(".W_K", ".k")
            path = path.replace(".W_V", ".v")
            path = path.replace(".b_Q", ".q")
            path = path.replace(".b_K", ".k")
            path = path.replace(".b_V", ".v")

        # Handle other attention weights
        path = path.replace(".W_O", ".o")
        path = path.replace(".b_O", ".o")

        # Handle MLP weights based on actual architecture
        # Check if this is an MLP weight that needs architecture-specific mapping
        if any(path.endswith(suffix) for suffix in [".W_in", ".W_out", ".b_in", ".b_out"]):
            # Extract the MLP component path (e.g., "blocks.0.mlp")
            mlp_path_parts = path.split(".")
            if len(mlp_path_parts) >= 3 and mlp_path_parts[-2] == "mlp":
                mlp_component_path = ".".join(mlp_path_parts[:-1])  # e.g., "blocks.0.mlp"

                # Check what MLP components are actually available
                try:
                    if self.component_mapping:
                        # Navigate to the MLP component to see what submodules it has
                        current_mapping = self.component_mapping
                        for part in mlp_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]

                        # Check available MLP subcomponents
                        if hasattr(current_mapping, "submodules"):
                            mlp_components = list(current_mapping.submodules.keys())

                            # Map based on available components
                            if "input" in mlp_components and "out" in mlp_components:
                                # GPT-2 style: input/out
                                path = path.replace(".W_in", ".input")
                                path = path.replace(".b_in", ".input")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "in" in mlp_components and "out" in mlp_components:
                                # Standard style: in/out
                                path = path.replace(".W_in", ".in")
                                path = path.replace(".b_in", ".in")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "fc_in" in mlp_components and "fc_out" in mlp_components:
                                # Some other style: fc_in/fc_out
                                path = path.replace(".W_in", ".fc_in")
                                path = path.replace(".b_in", ".fc_in")
                                path = path.replace(".W_out", ".fc_out")
                                path = path.replace(".b_out", ".fc_out")
                except Exception:
                    # Fallback to default behavior if component mapping inspection fails
                    pass

        # If no architecture-specific mapping was applied, use default fallback for MLP
        if any(path.endswith(suffix) for suffix in [".W_in", ".W_out", ".b_in", ".b_out"]):
            # Default fallback - assume standard in/out components
            path = path.replace(".W_in", ".in")
            path = path.replace(".b_in", ".in")
            path = path.replace(".W_out", ".out")
            path = path.replace(".b_out", ".out")
        path = path.replace(".W_gate", ".gate")
        path = path.replace(".b_gate", ".gate")

        # Handle embedding/unembedding weights (these keep their suffix)
        if not (path.endswith(".weight") or path.endswith(".bias")):
            path = path.replace(".W_E", "")
            path = path.replace(".b_E", "")
            path = path.replace(".W_U", "")
            path = path.replace(".b_U", "")
            path = path.replace(".W_pos", "")
            path = path.replace(".b_pos", "")
            path = path.replace(".w", "")
            path = path.replace(".b", "")

        return path, param_suffix

    def _translate_parameter_name(self, remote_path: str, original_path: str) -> str:
        """Translate parameter names from TransformerLens format to target format.

        Since preprocessing handles most parameter mapping, this method just
        handles any remaining cases.

        Args:
            remote_path: The translated component path
            original_path: The original TransformerLens path

        Returns:
            The path with parameter names translated
        """
        # Most parameter translation is handled by preprocessing,
        # so this method is now much simpler
        return remote_path

    def convert_weights(self, hf_model: nn.Module) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        if self.conversion_rules is None:
            raise ValueError("conversion_rules must be set before calling convert_weights")
        state_dict = self.conversion_rules.convert(input_value=hf_model)

        # Flatten state dictionary such that PyTorch can load it properly
        flattened_state_dict = self.flatten_nested_dict(state_dict)
        return flattened_state_dict

    def flatten_nested_dict(
        self,
        input: dict[str, torch.Tensor] | list[Any] | torch.Tensor,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, torch.Tensor]:
        """
        Flattens a nested dictionary/list structure into a flat dictionary with dot notation.

        Args:
            input: The input structure (can be dict, list, or a value)
            parent_key: The parent key for the current item (used in recursion)
            sep: Separator to use between nested keys (default '.')

        Returns:
            dict: Flattened dictionary with dot notation keys
        """
        items = {}

        if isinstance(input, dict):
            for k, v in input.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v

        elif isinstance(input, list):
            for i, v in enumerate(input):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = input

        return items
