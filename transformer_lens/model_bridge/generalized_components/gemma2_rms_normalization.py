"""Gemma2-specific RMS Normalization bridge component implementation.

Gemma2 uses a variant of RMSNorm where the weight is applied as (1 + weight) instead of just weight.
This is documented in https://github.com/huggingface/transformers/pull/29402
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.rms_normalization import (
    RMSNormalizationBridge,
)


class Gemma2RMSNormalizationBridge(RMSNormalizationBridge):
    """Gemma2-specific RMS Normalization bridge.

    Gemma2 uses a special RMSNorm formula:
        output = x * rsqrt(mean(x^2) + eps) * (1.0 + weight)

    This differs from standard RMSNorm which uses:
        output = x * rsqrt(mean(x^2) + eps) * weight

    The (1 + weight) formulation is specific to Gemma2 models.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the Gemma2 RMSNorm bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # keep mypy happy
        assert self.config is not None

        hidden_states = self.hook_in(hidden_states)

        # Check if we should use HuggingFace's autograd directly (for exact gradient matching)
        if self.use_native_layernorm_autograd:
            # Use HuggingFace's RMSNorm forward directly
            result = self._hf_autograd_forward(hidden_states)
        # Check if we should use LayerNormPre behavior (when layer norm folding is enabled)
        elif hasattr(self.config, "layer_norm_folding") and self.config.layer_norm_folding:
            # When folding is enabled, just do RMS normalization without weights
            # This is the "pre" version that just normalizes to unit scale
            # Note: Gemma2's (1 + weight) formula doesn't fold the same way as standard LayerNorm,
            # but for processed mode we skip the weight application entirely
            result = self._rmsnorm_pre_forward(hidden_states)
        else:
            # Gemma2-specific normalization behavior
            # Compute RMS (no centering for RMSNorm)
            scale = self.hook_scale(
                (
                    hidden_states.pow(2).mean(-1, keepdim=True) + getattr(self.config, "eps", 1e-5)
                ).sqrt()
            )

            # Match HookedTransformer's dtype casting after normalization
            dtype = getattr(self.config, "dtype", hidden_states.dtype)
            hidden_states = self.hook_normalized(hidden_states / scale).to(dtype)

            # Apply Gemma2's special (1 + weight) formula
            # This is the key difference from standard RMSNorm
            hidden_states = hidden_states * (1.0 + self.weight)

            result = hidden_states

        output = self.hook_out(result)
        return output

    def _hf_autograd_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delegating directly to HuggingFace's RMSNorm implementation.

        This ensures we match HF's computation exactly by delegating to the
        original component rather than reimplementing the logic.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        # Get parameters from the original component
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        # Simply delegate to the original HuggingFace component
        # This ensures exact numerical match including all dtype handling and computational graph
        return self.original_component(x)

    def _rmsnorm_pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for RMSNormPre behavior (normalization without weights).

        This is used when layer_norm_folding is enabled. It just normalizes
        to unit RMS scale without applying any learnable parameters.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        # Handle dtype conversion
        original_dtype = x.dtype
        config_dtype = getattr(self.config, "dtype", torch.float32)
        if config_dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        # RMSNorm: normalize without centering
        # Do NOT center for RMSNorm (no mean subtraction)
        eps = getattr(self.config, "eps", 1e-5)
        scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + eps).sqrt())
        result = self.hook_normalized(x / scale)

        # Convert back to original dtype
        return result.to(original_dtype)
