"""Neo architecture adapter."""

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
    PosEmbedBridge,
    UnembeddingBridge,
)


class NeoArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neo models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Neo architecture adapter."""
        super().__init__(cfg)

        # Ensure attention weights use split Q/K/V format when initializing the bridge
        self.cfg.split_attention_weights = True
        self.cfg.uses_split_attention = True

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "transformer.wte.weight",
                "pos_embed.pos": "transformer.wpe.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.attn.attention.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.attn.attention.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.attn.attention.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.attn.attention.out_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight.T",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight.T",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.u": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                # TransformerLens parameter mappings for processed weights
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.attention.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.attention.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.attention.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.attention.out_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.W_in": (
                    "transformer.h.{i}.mlp.c_fc.weight",
                    RearrangeHookConversion("o i -> i o"),
                ),
                "blocks.{i}.mlp.W_out": (
                    "transformer.h.{i}.mlp.c_proj.weight",
                    RearrangeHookConversion("o i -> i o"),
                ),
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attn.attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
