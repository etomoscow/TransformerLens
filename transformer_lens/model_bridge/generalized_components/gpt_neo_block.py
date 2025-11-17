"""GPT-Neo block bridge component.

This module contains a specialized bridge for GPT-Neo blocks that properly
handles residual stream hooks.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.block import BlockBridge


class GPTNeoBlockBridge(BlockBridge):
    """Specialized BlockBridge for GPT-Neo models.

    GPT-Neo blocks have a specific residual connection pattern that requires
    custom hook placement to match HookedTransformer behavior:
    1. resid_pre (input to block)
    2. ln1 + attention
    3. resid_mid = resid_pre + attn_out  ← hook fires here
    4. ln2 + MLP
    5. resid_post = resid_mid + mlp_out  ← hook fires here
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GPT-Neo block bridge.

        Args:
            name: The name of this component
            config: Model configuration
            submodules: Dictionary of submodules (ln1, attn, ln2, mlp)
        """
        super().__init__(name, config, submodules=submodules)

        # Create actual hook_resid_mid instead of just aliasing
        self.hook_resid_mid = HookPoint()
        self._register_hook("hook_resid_mid", self.hook_resid_mid)

        # Update hook aliases to NOT alias hook_resid_mid since we have a real one
        # Remove the alias so it uses the real hook instead
        if "hook_resid_mid" in self.hook_aliases:
            del self.hook_aliases["hook_resid_mid"]

        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module):
        """Set the original component and monkey-patch its forward method.

        Args:
            component: The original PyTorch module to wrap
        """
        super().set_original_component(component)
        self._patch_gpt_neo_block_forward()

    def _patch_gpt_neo_block_forward(self):
        """Monkey-patch the GPT-Neo block's forward method to insert hooks."""
        if self.original_component is None:
            return

        # Save the original forward method
        self._original_block_forward = self.original_component.forward

        def patched_forward(
            block_self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
            cache_position=None,
        ):
            """Patched GPT-Neo block forward with residual hooks.

            This reimplements the GPTNeoBlock forward pass with hooks at the
            correct positions to match HookedTransformer behavior.

            Note: We use the bridge's wrapped components (self.ln1, self.attn, etc.)
            instead of the HF block's components to ensure hooks fire correctly.
            """
            # hook_resid_pre fires on block input via self.hook_in (from base class)
            residual = hidden_states

            # Use bridge's wrapped components
            if hasattr(self, "ln1"):
                hidden_states = self.ln1(hidden_states)
            else:
                hidden_states = block_self.ln_1(hidden_states)

            if hasattr(self, "attn"):
                attn_result = self.attn(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
                # AttentionBridge returns tuple (output, weights)
                if isinstance(attn_result, tuple):
                    attn_output, attn_weights = attn_result
                else:
                    attn_output = attn_result
                    attn_weights = None
            else:
                attn_output, attn_weights = block_self.attn(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )

            # First residual connection - hook_resid_mid fires here
            hidden_states = attn_output + residual
            hidden_states = self.hook_resid_mid(hidden_states)

            # Second stage
            residual = hidden_states
            if hasattr(self, "ln2"):
                hidden_states = self.ln2(hidden_states)
            else:
                hidden_states = block_self.ln_2(hidden_states)

            if hasattr(self, "mlp"):
                feed_forward_hidden_states = self.mlp(hidden_states)
            else:
                feed_forward_hidden_states = block_self.mlp(hidden_states)

            # Second residual connection
            hidden_states = residual + feed_forward_hidden_states
            # hook_resid_post fires on block output via self.hook_out (from base class)

            return hidden_states, attn_weights

        # Bind the patched method to self (the bridge), not block_self
        self.original_component.forward = patched_forward.__get__(self, type(self))

    def __repr__(self) -> str:
        """String representation of the GPTNeoBlockBridge."""
        return f"GPTNeoBlockBridge(name={self.name})"
