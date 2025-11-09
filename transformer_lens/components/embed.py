"""Hooked Transformer Embed Component.

This module contains all the component :class:`Embed`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import LayerNorm
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if self.cfg.post_embedding_ln:
            self.ln = LayerNorm(self.cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        embed_out = self.W_E[tokens, :]

        # Apply post-embedding scaling if configured (e.g., Gemma models scale by sqrt(d_model))
        if self.cfg.post_embedding_scale is not None:
            embed_out = embed_out * self.cfg.post_embedding_scale

        # Apply post-embedding layer norm if configured (e.g., Bloom)
        if self.cfg.post_embedding_ln:
            embed_out = self.ln(embed_out)

        return embed_out
