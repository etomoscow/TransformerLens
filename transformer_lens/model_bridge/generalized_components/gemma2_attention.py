"""Gemma2-specific Attention bridge component implementation.

Gemma2 uses the same position embeddings system as Gemma3 but needs special handling
for Q/K/V hook shapes to match HookedTransformer's expectations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.model_bridge.generalized_components.gemma3_attention import (
    PositionEmbeddingsAttentionBridge,
)


class Gemma2AttentionBridge(PositionEmbeddingsAttentionBridge):
    """Gemma2-specific Attention bridge with Q/K/V hook reshaping.

    Gemma2 needs Q/K/V hooks to output separated heads for HookedTransformer compatibility.
    HuggingFace outputs combined heads [batch, seq, d_proj], but HookedTransformer expects
    [batch, seq, n_heads, d_head].
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize Gemma2 attention bridge.

        Args:
            name: Component name
            config: Model configuration
            submodules: Dictionary of subcomponents
            **kwargs: Additional arguments passed to PositionEmbeddingsAttentionBridge
        """
        super().__init__(name, config, submodules, **kwargs)

        # Note: Q/K/V hook conversions will be set up in set_original_component
        # after submodules are properly initialized

    def _setup_qkv_hook_conversions(self) -> None:
        """Setup Q/K/V hooks to reshape from combined to separated heads."""

        if self.config is None:
            return  # Skip if no config available

        # Get model dimensions
        n_heads = self.config.n_heads if hasattr(self.config, "n_heads") else 8
        n_kv_heads = self.config.n_key_value_heads if hasattr(self.config, "n_key_value_heads") else n_heads

        # Q projection: 2304 -> 2048 (8 heads * 256)
        # K/V projection: 2304 -> 1024 (4 heads * 256)
        d_head = 256  # Gemma2 uses fixed head dimension of 256

        class ReshapeQKVHeads(BaseHookConversion):
            """Reshape Q/K/V tensors to split attention heads."""

            def __init__(self, n_heads: int, d_head: int):
                super().__init__()
                self.n_heads = n_heads
                self.d_head = d_head

            def handle_conversion(self, input_value, *full_context):
                """Convert from [batch, seq, n_heads*d_head] to [batch, seq, n_heads, d_head]."""
                if len(input_value.shape) == 3:
                    b, s, d = input_value.shape
                    expected_d = self.n_heads * self.d_head
                    if d == expected_d:
                        return input_value.view(b, s, self.n_heads, self.d_head)
                return input_value

            def revert(self, input_value, *full_context):
                """Revert from [batch, seq, n_heads, d_head] to [batch, seq, n_heads*d_head]."""
                if len(input_value.shape) == 4:
                    b, s, n_h, d_h = input_value.shape
                    if n_h == self.n_heads and d_h == self.d_head:
                        return input_value.view(b, s, n_h * d_h)
                return input_value

        # Apply conversions to Q/K/V hooks if they exist
        if 'q' in self.submodules and hasattr(self.submodules['q'], 'hook_out'):
            q_reshape = ReshapeQKVHeads(n_heads, d_head)
            self.submodules['q'].hook_out.hook_conversion = q_reshape

        if 'k' in self.submodules and hasattr(self.submodules['k'], 'hook_out'):
            k_reshape = ReshapeQKVHeads(n_kv_heads, d_head)
            self.submodules['k'].hook_out.hook_conversion = k_reshape

        if 'v' in self.submodules and hasattr(self.submodules['v'], 'hook_out'):
            v_reshape = ReshapeQKVHeads(n_kv_heads, d_head)
            self.submodules['v'].hook_out.hook_conversion = v_reshape

    def set_original_component(self, component: Any) -> None:
        """Set the original HuggingFace component and apply Q/K/V hook conversions.

        Args:
            component: The original HuggingFace attention component
        """
        super().set_original_component(component)
        # Apply hook conversions after component is set and submodules are ready
        self._setup_qkv_hook_conversions()

    def setup_for_transformerbridge(self) -> None:
        """Called after TransformerBridge initialization to set up hook conversions."""
        super().setup_for_transformerbridge()
        # Ensure hook conversions are applied
        self._setup_qkv_hook_conversions()