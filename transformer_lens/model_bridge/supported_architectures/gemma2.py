"""Gemma2 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.gemma2_embedding import (
    Gemma2EmbeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.gemma2_rms_normalization import (
    Gemma2RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.gemma2_attention import (
    Gemma2AttentionBridge,
)


class Gemma2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma2 architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"  # Gemma-2 uses RMSNorm, not LayerNorm

        # Use SDPA for numerical consistency in attention components
        if self.cfg.attn_implementation is None:
            self.cfg.attn_implementation = "sdpa"

        # Note: n_key_value_heads is now automatically mapped from num_key_value_heads
        # by map_default_transformer_lens_config() in sources/transformers.py

        self.conversion_rules = HookConversionSet(
            {
                # Note: Gemma2 scales embeddings by sqrt(d_model) in HookedTransformer
                # but NOT in HuggingFace. Since Bridge wraps HF, it also doesn't scale.
                # This creates a mismatch in Phase 2/3 that we accept as a known limitation.
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1_post.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.pre_feedforward_layernorm.weight",
                "blocks.{i}.ln2_post.w": "model.layers.{i}.post_feedforward_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight.T",  # Not shared with embedding
            }
        )

        self.component_mapping = {
            # Use Gemma2EmbeddingBridge with hook conversion for Phase 2/3 scaling
            # It returns unscaled embeddings (matching HF) but applies scaling via hook conversion
            "embed": Gemma2EmbeddingBridge(name="model.embed_tokens", config=self.cfg),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": Gemma2RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": Gemma2RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln2": Gemma2RMSNormalizationBridge(
                        name="pre_feedforward_layernorm", config=self.cfg
                    ),
                    "ln2_post": Gemma2RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
                    ),
                    "attn": Gemma2AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": Gemma2RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Gemma-2 component testing.

        Gemma-2 uses standard RoPE. We set the rotary_emb on all attention bridge
        instances for component testing.

        We also enable use_native_layernorm_autograd on all normalization bridges to ensure
        we match HuggingFace's numerical outputs exactly by delegating to their implementation.

        Args:
            hf_model: The HuggingFace Gemma-2 model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instance from the model
        rotary_emb = hf_model.model.rotary_emb

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, 'blocks'):
            # Set on each layer's actual attention bridge instance
            for block in bridge_model.blocks:
                if hasattr(block, 'attn'):
                    block.attn.set_rotary_emb(rotary_emb)

                # Enable native HF autograd for all normalization layers
                if hasattr(block, 'ln1'):
                    block.ln1.use_native_layernorm_autograd = True
                if hasattr(block, 'ln1_post'):
                    block.ln1_post.use_native_layernorm_autograd = True
                if hasattr(block, 'ln2'):
                    block.ln2.use_native_layernorm_autograd = True
                if hasattr(block, 'ln2_post'):
                    block.ln2_post.use_native_layernorm_autograd = True

            # Enable for final layer norm
            if hasattr(bridge_model, 'ln_final'):
                bridge_model.ln_final.use_native_layernorm_autograd = True

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
