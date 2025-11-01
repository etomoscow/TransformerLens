"""GPT-OSS architecture adapter."""

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
    JointGateUpMLPBridge,
    LinearBridge,
    NormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class GPTOSSArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-OSS model."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT-OSS architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        # GPT-OSS uses 'variance_epsilon' instead of 'eps' for RMSNorm
        self.cfg.eps_attr = "variance_epsilon"

        # Conversion rules for weight processing/folding
        # GPT-OSS uses MoE with batched experts, so we need special handling
        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                # Note: MLP weights for MoE models with batched experts are not directly mappable
                # The experts use batched tensors [num_experts, ...] which need special handling
                # These mappings are for the router only
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight.T",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": JointGateUpMLPBridge(
                        name="mlp.experts",
                        gate_up_config={"split_gate_up_matrix": self.split_gate_up_matrix},
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def split_gate_up_matrix(
        self, original_mlp_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear]:
        gate_up_weight = original_mlp_component.gate_up_proj
        gate_up_bias = original_mlp_component.gate_up_proj_bias

        # In GPT-OSS, all the gate projection weights lie at even indices,
        # all the up projection weights lie at odd indices
        gate_weight = gate_up_weight[..., ::2]
        up_weight = gate_up_weight[..., 1::2]

        gate_bias = gate_up_bias[..., ::2]
        up_bias = gate_up_bias[..., 1::2]

        gate_projection = torch.nn.Linear(gate_weight.shape[0], gate_weight.shape[1], bias=True)

        gate_projection.weight = torch.nn.Parameter(gate_weight)
        gate_projection.bias = torch.nn.Parameter(gate_bias)

        up_projection = torch.nn.Linear(up_weight.shape[0], up_weight.shape[1])

        up_projection.weight = torch.nn.Parameter(up_weight)
        up_projection.bias = torch.nn.Parameter(up_bias)

        return gate_projection, up_projection
