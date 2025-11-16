"""Neo architecture adapter."""

from typing import Any

import torch

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
from transformer_lens.model_bridge.generalized_components.gpt_neo_block import (
    GPTNeoBlockBridge,
)


class NeoArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neo models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Neo architecture adapter."""
        super().__init__(cfg)

        # GPT-Neo uses BOS tokens (inherits default_prepend_bos = True)

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
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.u": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),
            "blocks": GPTNeoBlockBridge(
                name="transformer.h",
                config=self.cfg,
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

    def preprocess_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Transpose GPT-Neo Linear MLP weights to Conv1D format for weight processing.

        GPT-Neo uses standard PyTorch Linear layers with weights shaped [out_features, in_features].
        However, the weight processing code (fold_layer_norm, etc.) expects Conv1D format
        with weights shaped [in_features, out_features].

        This method transposes MLP weights to match the expected format:
        - c_fc.weight: [3072, 768] -> [768, 3072]  (d_mlp, d_model) -> (d_model, d_mlp)
        - c_proj.weight: [768, 3072] -> [3072, 768]  (d_model, d_mlp) -> (d_mlp, d_model)

        Args:
            state_dict: The state dictionary with HuggingFace format keys and Linear weights

        Returns:
            The modified state dictionary with transposed MLP weights in Conv1D format
        """
        n_layers = self.cfg.n_layers if hasattr(self.cfg, "n_layers") else self.cfg.num_layers

        for layer_idx in range(n_layers):
            # Transpose MLP input weight from Linear [d_mlp, d_model] to Conv1D [d_model, d_mlp]
            c_fc_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            if c_fc_key in state_dict:
                state_dict[c_fc_key] = state_dict[c_fc_key].T

            # Transpose MLP output weight from Linear [d_model, d_mlp] to Conv1D [d_mlp, d_model]
            c_proj_key = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
            if c_proj_key in state_dict:
                state_dict[c_proj_key] = state_dict[c_proj_key].T

        return state_dict
