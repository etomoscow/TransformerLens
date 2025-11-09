"""Gemma2-specific Embedding bridge component implementation.

Gemma2 embeddings need to be scaled by sqrt(d_model).
HuggingFace's Gemma2 implementation returns unscaled embeddings.
"""

from typing import Any, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.model_bridge.generalized_components.embedding import (
    EmbeddingBridge,
)


class EmbeddingScaleConversion(BaseHookConversion):
    """Hook conversion to scale embeddings by sqrt(d_model) for Gemma2."""

    def __init__(self, scale_factor: float):
        """Initialize with the scale factor.

        Args:
            scale_factor: The factor to scale embeddings by (sqrt(d_model))
        """
        super().__init__()
        self.scale_factor = scale_factor

    def handle_conversion(self, input_value, *full_context):
        """Scale the embeddings when comparing with HookedTransformer."""
        return input_value * self.scale_factor

    def revert(self, input_value, *full_context):
        """Revert the scaling when going back."""
        return input_value / self.scale_factor


class Gemma2EmbeddingBridge(EmbeddingBridge):
    """Gemma2-specific Embedding bridge.

    Gemma2 requires embeddings to be scaled by sqrt(d_model) when comparing
    with HookedTransformer, but NOT when comparing with HuggingFace.

    This bridge returns unscaled embeddings (matching HF) but applies a
    hook conversion to scale them when comparing with HookedTransformer.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the Gemma2 embedding bridge.

        Args:
            name: Component name
            config: Model configuration
            submodules: Dictionary of subcomponents
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(name, config, submodules, **kwargs)

        # Apply scaling hook conversion if config is available
        if config is not None and hasattr(config, 'd_model'):
            scale_factor = config.d_model ** 0.5
            # Apply conversion to the hook_out for Phase 2/3 comparisons
            if hasattr(self, 'hook_out'):
                self.hook_out.hook_conversion = EmbeddingScaleConversion(scale_factor)
