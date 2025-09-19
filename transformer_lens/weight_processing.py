#!/usr/bin/env python3
"""
Weight Processing Functions for Transformer Models.

This module contains all the weight processing functions extracted from HookedTransformer,
organized into a single ProcessWeights class with static methods. These functions are used
to modify transformer model weights for better interpretability and analysis.
"""

from typing import Dict

import einops
import torch
from torch import nn

import transformer_lens.utils as utils
from transformer_lens.FactoredMatrix import FactoredMatrix


class ProcessWeights:
    """
    A collection of static methods for processing transformer model weights.

    These methods are extracted from HookedTransformer and provide various weight
    transformations for improved model interpretability:
    - LayerNorm folding: Merges LayerNorm parameters into subsequent linear layers
    - Weight centering: Centers weights that write to the residual stream
    - Unembed centering: Centers unembedding weights (translation invariant)
    - Value bias folding: Consolidates value biases into output biases
    - Attention matrix refactoring: Experimental QK/OV matrix factorization

    When an architecture adapter is provided, the methods will translate TransformerLens
    parameter names to the target format (e.g., HuggingFace) for processing.
    """

    @staticmethod
    def _get_param_key(tl_key: str, adapter=None) -> str:
        """Get the actual parameter key to use, translating via adapter if provided.

        Args:
            tl_key: TransformerLens format parameter key
            adapter: Optional architecture adapter for key translation

        Returns:
            The key to use for accessing parameters in the state dict
        """
        if adapter is None:
            return tl_key

        # Use the adapter to translate from TL format to target format
        return adapter.translate_transformer_lens_path(tl_key)

    @staticmethod
    def _fold_layer(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        layer_idx: int,
        fold_biases: bool,
        center_weights: bool,
        adapter,
        gqa: str,
    ) -> None:
        """Fold LayerNorm for a single layer.
        
        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            layer_idx: The layer index to process
            fold_biases: Whether to fold LayerNorm biases
            center_weights: Whether to center weights after folding
            adapter: Optional architecture adapter for parameter key translation
            gqa: GQA prefix string (empty or "_")
        """
        l = layer_idx
        
        # When adapter is provided, always use HuggingFace format processing
        if adapter:
            # HuggingFace format: combined QKV weights and biases
            # Process the combined QKV tensor once
            W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
            b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
            ln1_b_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.b", adapter)
            ln1_w_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.w", adapter)
            
            # Get the combined QKV tensors
            qkv_weight = state_dict[W_Q_key]  # [d_model, 3 * d_model]
            qkv_bias = state_dict[b_Q_key]    # [3 * n_heads * d_head]
            ln1_b = state_dict[ln1_b_key]     # [d_model]
            ln1_w = state_dict[ln1_w_key]     # [d_model]
            
            n_heads = cfg.n_heads
            d_head = cfg.d_head
            d_model = cfg.d_model
            
            # Split the combined QKV weight and bias
            W_Q, W_K, W_V = torch.tensor_split(qkv_weight, 3, dim=1)  # Each: [d_model, d_model]
            b_Q, b_K, b_V = torch.tensor_split(qkv_bias, 3, dim=0)    # Each: [n_heads * d_head]
            
            # Fold layer norm into QKV weights and biases
            # COMMENTED OUT: Bias folding to find specific problematic lines
            if fold_biases:
                # Fold ln1 bias into QKV biases
                # For each of Q, K, V: b_new = b_old + (W * ln_b).sum(dim=0)
                b_Q_new = b_Q + (W_Q * ln1_b[:, None]).sum(dim=0)
                b_K_new = b_K + (W_K * ln1_b[:, None]).sum(dim=0)
                b_V_new = b_V + (W_V * ln1_b[:, None]).sum(dim=0)
                
                # Combine back into single QKV bias
                new_qkv_bias = torch.cat([b_Q_new, b_K_new, b_V_new])
                state_dict[b_Q_key] = new_qkv_bias
                del state_dict[ln1_b_key]
            
            # Fold ln1 weight into QKV weights
            W_Q_new = W_Q * ln1_w[:, None]
            W_K_new = W_K * ln1_w[:, None]
            W_V_new = W_V * ln1_w[:, None]
            
            # Combine back into single QKV weight
            new_qkv_weight = torch.cat([W_Q_new, W_K_new, W_V_new], dim=1)
            state_dict[W_Q_key] = new_qkv_weight
            del state_dict[ln1_w_key]
            
            # Center the weights if requested
            if center_weights:
                # Center each of Q, K, V weights along the d_model dimension (dim=0)
                # This matches the TransformerLens format centering: "head_index d_model d_head -> head_index 1 d_head"
                # For HuggingFace format [d_model, d_model], we center along dim=0 (d_model)
                W_Q_centered = W_Q_new - W_Q_new.mean(dim=0, keepdim=True)
                W_K_centered = W_K_new - W_K_new.mean(dim=0, keepdim=True)
                W_V_centered = W_V_new - W_V_new.mean(dim=0, keepdim=True)
                
                # Combine back into single QKV weight
                centered_qkv_weight = torch.cat([W_Q_centered, W_K_centered, W_V_centered], dim=1)
                state_dict[W_Q_key] = centered_qkv_weight
                
        else:
            # TransformerLens format: separate Q, K, V weights and biases
            # Get translated parameter keys for separate format
            b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
            W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
            b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}b_K", adapter)
            W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}W_K", adapter)
            b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}b_V", adapter)
            W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}W_V", adapter)
            ln1_b_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.b", adapter)
            ln1_w_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.w", adapter)
            
            # Check if LayerNorm parameters exist (they might not for already processed models)
            if ln1_b_key in state_dict and ln1_w_key in state_dict:
                # Fold ln1 into attention - it's important to fold biases first, since biases depend on
                # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
                # along every axis other than d_model. Each weight matrix right multiplies. To fold in
                # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
                # to sum along axis -2, which is the residual stream space axis.
                if fold_biases:
                    state_dict[b_Q_key] = state_dict[b_Q_key] + (
                        state_dict[W_Q_key] * state_dict[ln1_b_key][None, :, None]
                    ).sum(-2)
                    state_dict[b_K_key] = state_dict[b_K_key] + (
                        state_dict[W_K_key] * state_dict[ln1_b_key][None, :, None]
                    ).sum(-2)
                    state_dict[b_V_key] = state_dict[b_V_key] + (
                        state_dict[W_V_key] * state_dict[ln1_b_key][None, :, None]
                    ).sum(-2)
                    del state_dict[ln1_b_key]

                state_dict[W_Q_key] = state_dict[W_Q_key] * state_dict[ln1_w_key][None, :, None]
                state_dict[W_K_key] = state_dict[W_K_key] * state_dict[ln1_w_key][None, :, None]
                state_dict[W_V_key] = state_dict[W_V_key] * state_dict[ln1_w_key][None, :, None]
                del state_dict[ln1_w_key]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[W_Q_key] -= einops.reduce(
                    state_dict[W_Q_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[W_K_key] -= einops.reduce(
                    state_dict[W_K_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[W_V_key] -= einops.reduce(
                    state_dict[W_V_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

        # Fold ln2 into MLP
        if not getattr(cfg, "attn_only", False):
            # Get translated MLP parameter keys
            mlp_b_in_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_in", adapter)
            mlp_W_in_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_in", adapter)

            # Only get gate key if model has gated MLPs
            mlp_W_gate_key = None
            if getattr(cfg, "gated_mlp", False):
                mlp_W_gate_key = ProcessWeights._get_param_key(
                    f"blocks.{l}.mlp.W_gate", adapter
                )

            ln2_b_key = ProcessWeights._get_param_key(f"blocks.{l}.ln2.b", adapter)
            ln2_w_key = ProcessWeights._get_param_key(f"blocks.{l}.ln2.w", adapter)

            # Check if MLP LayerNorm parameters exist (they might not for already processed models)
            if ln2_b_key in state_dict and ln2_w_key in state_dict:
                if fold_biases:
                    state_dict[mlp_b_in_key] = state_dict[mlp_b_in_key] + (
                        state_dict[mlp_W_in_key] * state_dict[ln2_b_key][:, None]
                    ).sum(-2)
                    del state_dict[ln2_b_key]

                state_dict[mlp_W_in_key] = state_dict[mlp_W_in_key] * state_dict[ln2_w_key][:, None]

                if getattr(cfg, "gated_mlp", False) and mlp_W_gate_key is not None:
                    state_dict[mlp_W_gate_key] = (
                        state_dict[mlp_W_gate_key] * state_dict[ln2_w_key][:, None]
                    )

                del state_dict[ln2_w_key]

            if center_weights:
                # Center the weights that read in from the LayerNormPre
                state_dict[mlp_W_in_key] -= einops.reduce(
                    state_dict[mlp_W_in_key],
                    "d_model d_mlp -> 1 d_mlp",
                    "mean",
                )

            if getattr(cfg, "act_fn", None) is not None and cfg.act_fn.startswith("solu"):
                # Get translated SoLU LayerNorm parameter keys
                mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)
                mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
                mlp_ln_b_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.ln.b", adapter)
                mlp_ln_w_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.ln.w", adapter)

                # Fold ln3 into activation
                if fold_biases:
                    state_dict[mlp_b_out_key] = state_dict[mlp_b_out_key] + (
                        state_dict[mlp_W_out_key] * state_dict[mlp_ln_b_key][:, None]
                    ).sum(-2)

                    del state_dict[mlp_ln_b_key]

                state_dict[mlp_W_out_key] = (
                    state_dict[mlp_W_out_key] * state_dict[mlp_ln_w_key][:, None]
                )

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[mlp_W_out_key] -= einops.reduce(
                        state_dict[mlp_W_out_key],
                        "d_mlp d_model -> 1 d_model",
                        "mean",
                    )

                del state_dict[mlp_ln_w_key]

    @staticmethod
    def fold_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_biases: bool = True,
        center_weights: bool = True,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            cfg: Model configuration object with n_layers, n_key_value_heads, etc.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with LayerNorm folded into linear layers.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if getattr(cfg, "n_key_value_heads", None) is None else "_"

        print("n layers", cfg.n_layers)
        for l in range(cfg.n_layers):
            ProcessWeights._fold_layer(
                state_dict, cfg, l, fold_biases, center_weights, adapter, gqa
            )

        # Fold ln_final into Unembed
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
        ln_final_w_key = ProcessWeights._get_param_key("ln_final.w", adapter)

        # Check if unembedding bias actually exists (some models like GPT-2 don't have it)
        has_unembed_bias = unembed_b_U_key in state_dict

        if not getattr(cfg, "final_rms", False) and fold_biases and has_unembed_bias:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            unembed_weight = state_dict[unembed_W_U_key]
            ln_bias = state_dict[ln_final_b_key]
            
            # Handle different tensor shapes for bias folding
            if len(unembed_weight.shape) == 2 and len(ln_bias.shape) == 1:
                if unembed_weight.shape[1] == ln_bias.shape[0]:
                    # HuggingFace format: [vocab_size, d_model] * [d_model] -> sum over d_model
                    bias_contribution = (unembed_weight * ln_bias[None, :]).sum(dim=-1)
                elif unembed_weight.shape[0] == ln_bias.shape[0]:
                    # TransformerLens format: [d_model, vocab_size] * [d_model] -> sum over d_model
                    bias_contribution = (unembed_weight * ln_bias[:, None]).sum(dim=-2)
                else:
                    raise ValueError(f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm bias {ln_bias.shape}")
            else:
                raise ValueError(f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm bias {ln_bias.shape}")
            
            state_dict[unembed_b_U_key] = state_dict[unembed_b_U_key] + bias_contribution
            del state_dict[ln_final_b_key]

        # Generalized layer norm folding for unembedding
        unembed_weight = state_dict[unembed_W_U_key]
        ln_weight = state_dict[ln_final_w_key]
        
        # Handle different tensor shapes (TransformerLens vs HuggingFace format)
        if len(unembed_weight.shape) == 2 and len(ln_weight.shape) == 1:
            # Check if we need to transpose for proper broadcasting
            if unembed_weight.shape[1] == ln_weight.shape[0]:
                # HuggingFace format: [vocab_size, d_model] * [d_model] -> [vocab_size, d_model]
                state_dict[unembed_W_U_key] = unembed_weight * ln_weight[None, :]
            elif unembed_weight.shape[0] == ln_weight.shape[0]:
                # TransformerLens format: [d_model, vocab_size] * [d_model] -> [d_model, vocab_size]
                state_dict[unembed_W_U_key] = unembed_weight * ln_weight[:, None]
            else:
                raise ValueError(f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm weight {ln_weight.shape}")
        else:
            raise ValueError(f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm {ln_weight.shape}")
        del state_dict[ln_final_w_key]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            unembed_weight = state_dict[unembed_W_U_key]
            if len(unembed_weight.shape) == 2:
                if unembed_weight.shape[0] > unembed_weight.shape[1]:
                    # TransformerLens format: [d_model, vocab_size] - center along d_model
                    state_dict[unembed_W_U_key] -= einops.reduce(
                        unembed_weight, "d_model d_vocab -> 1 d_vocab", "mean"
                    )
                else:
                    # HuggingFace format: [vocab_size, d_model] - center along d_model
                    state_dict[unembed_W_U_key] -= einops.reduce(
                        unembed_weight, "vocab_size d_model -> vocab_size 1", "mean"
                    )
            else:
                raise ValueError(f"Unexpected unembedding weight shape: {unembed_weight.shape}")

        return state_dict

    @staticmethod
    def center_writing_weights(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered writing weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Get translated parameter keys
        embed_W_E_key = ProcessWeights._get_param_key("embed.W_E", adapter)
        pos_embed_W_pos_key = ProcessWeights._get_param_key("pos_embed.W_pos", adapter)

        state_dict[embed_W_E_key] = state_dict[embed_W_E_key] - state_dict[embed_W_E_key].mean(
            -1, keepdim=True
        )
        if getattr(cfg, "positional_embedding_type", "standard") != "rotary":
            state_dict[pos_embed_W_pos_key] = state_dict[pos_embed_W_pos_key] - state_dict[
                pos_embed_W_pos_key
            ].mean(-1, keepdim=True)

        for l in range(cfg.n_layers):
            # Get translated parameter keys for this layer
            attn_W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
            attn_b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
            mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
            mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)

            state_dict[attn_W_O_key] = state_dict[attn_W_O_key] - state_dict[attn_W_O_key].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[attn_b_O_key] = (
                state_dict[attn_b_O_key] - state_dict[attn_b_O_key].mean()
            )  # b_O is [d_model]
            if not getattr(cfg, "attn_only", False):
                state_dict[mlp_W_out_key] = state_dict[mlp_W_out_key] - state_dict[
                    mlp_W_out_key
                ].mean(-1, keepdim=True)
                state_dict[mlp_b_out_key] = (
                    state_dict[mlp_b_out_key] - state_dict[mlp_b_out_key].mean()
                )
        return state_dict

    @staticmethod
    def center_unembed(
        state_dict: Dict[str, torch.Tensor], adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center the unembedding weights W_U.

        This is done by subtracting the mean of the weights from the weights themselves. This is
        done in-place. As softmax is translation invariant, this changes the logits but not the log
        probs, and makes the model logits (slightly) more interpretable - when trying to understand
        how components contribute to the logits, we'll be less misled by components that just add
        something to every logit.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered unembedding weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Get translated parameter keys
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)

        state_dict[unembed_W_U_key] = state_dict[unembed_W_U_key] - state_dict[
            unembed_W_U_key
        ].mean(-1, keepdim=True)

        # Only center bias if it exists (some models like GPT-2 don't have unembedding bias)
        if unembed_b_U_key in state_dict:
            state_dict[unembed_b_U_key] = (
                state_dict[unembed_b_U_key] - state_dict[unembed_b_U_key].mean()
            )
        return state_dict

    @staticmethod
    def fold_value_biases(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with value biases folded into output bias.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        for layer in range(cfg.n_layers):
            # Get translated parameter keys
            if getattr(cfg, "n_key_value_heads", None) is None:
                b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
            else:
                b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn._b_V", adapter)

            # Get other translated parameter keys
            W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
            b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)

            # Check if we have combined QKV format (HuggingFace) or separate format (TransformerLens)
            if b_V_key in state_dict:
                b_V = state_dict[b_V_key]
                W_O = state_dict[W_O_key]
                b_O_original = state_dict[b_O_key]
                
                # Handle different tensor formats
                if len(b_V.shape) == 1 and len(W_O.shape) == 2:
                    # HuggingFace format: combined QKV bias [3 * n_heads * d_head], W_O [d_model, d_model]
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model
                    
                    # Extract just the V bias from the combined QKV bias
                    # Combined bias is [Q_bias, K_bias, V_bias] where each is [n_heads * d_head]
                    v_bias_start = 2 * n_heads * d_head  # Start of V bias
                    v_bias_end = 3 * n_heads * d_head    # End of V bias
                    b_V_only = b_V[v_bias_start:v_bias_end]  # [n_heads * d_head]
                    
                    # Reshape for computation: [n_heads * d_head] -> [n_heads, d_head]
                    b_V_reshaped = b_V_only.reshape(n_heads, d_head)
                    
                    # W_O is [d_model, d_model], we need to reshape it to [n_heads, d_head, d_model]
                    # W_O represents the output projection, so we need to split it by heads
                    W_O_reshaped = W_O.T.reshape(n_heads, d_head, d_model)
                    
                    # Compute the folded bias: sum over heads and d_head dimensions
                    folded_b_O = b_O_original + (b_V_reshaped[:, :, None] * W_O_reshaped).sum([0, 1])
                    
                    # Zero out the V bias in the combined QKV bias
                    new_b_V = b_V.clone()
                    new_b_V[v_bias_start:v_bias_end] = 0
                    state_dict[b_V_key] = new_b_V
                    
                elif len(b_V.shape) == 2 and len(W_O.shape) == 3:
                    # TransformerLens format: separate V bias [n_heads, d_head], W_O [n_heads, d_head, d_model]
                    if getattr(cfg, "n_key_value_heads", None) is not None:
                        b_V = torch.repeat_interleave(
                            b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                        )
                    
                    folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])
                    state_dict[b_V_key] = torch.zeros_like(b_V)
                else:
                    raise ValueError(f"Unexpected tensor shapes: b_V {b_V.shape}, W_O {W_O.shape}")
                
                state_dict[b_O_key] = folded_b_O
                
        return state_dict

    @staticmethod
    def refactor_factored_attn_matrices(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with refactored attention matrices.
        """
        assert (
            getattr(cfg, "positional_embedding_type", "standard") != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        for l in range(cfg.n_layers):
            # Get translated parameter keys
            W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
            b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
            W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_K", adapter)
            b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_K", adapter)
            W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_V", adapter)
            W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
            b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_V", adapter)
            b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)

            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[W_Q_key],
                    state_dict[b_Q_key][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[W_K_key],
                    state_dict[b_K_key][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[W_Q_key] = W_Q_eff_even[:, :-1, :]
            state_dict[b_Q_key] = W_Q_eff_even[:, -1, :]
            state_dict[W_K_key] = W_K_eff_even[:, :-1, :]
            state_dict[b_K_key] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[W_V_key]
            W_O = state_dict[W_O_key]

            # Factors the bias to be consistent.
            b_V = state_dict[b_V_key]
            b_O = state_dict[b_O_key]

            # Add singleton dimension for broadcasting
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

            # Element-wise multiplication of b_V and W_O
            b_V_times_W_O = b_V_expanded * W_O

            # Sum over d_head and head_index dimensions
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)

            effective_bias = b_O + b_V_contribution
            state_dict[b_V_key] = torch.zeros_like(b_V)
            state_dict[b_O_key] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[W_V_key] = U @ S.diag_embed()
            state_dict[W_O_key] = utils.transpose(Vh)

        return state_dict

    @staticmethod
    def process_weights(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Apply all weight processing transformations in the correct order.

        This is a convenience function that applies all the weight processing steps
        in the same order as HookedTransformer.load_and_process_state_dict().

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            fold_ln (bool): Whether to fold LayerNorm weights into subsequent layers.
            center_writing_weights (bool): Whether to center weights writing to residual stream.
            center_unembed (bool): Whether to center unembedding weights.
            fold_value_biases (bool): Whether to fold value biases into output bias.
            refactor_factored_attn_matrices (bool): Whether to refactor attention matrices.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Fully processed state dict.
        """
        processed_dict = state_dict.copy()

        if fold_ln:
            if getattr(cfg, "num_experts", None) and cfg.num_experts > 1:
                # Skip for MoE models
                pass
            elif getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, fold_biases=True, center_weights=True, adapter=adapter
                )
            elif getattr(cfg, "normalization_type", "LN") in ["RMS", "RMSPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, fold_biases=False, center_weights=False, adapter=adapter
                )

        if center_writing_weights:
            if getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"] and not getattr(
                cfg, "final_rms", False
            ):
                processed_dict = ProcessWeights.center_writing_weights(
                    processed_dict, cfg, adapter=adapter
                )

        if center_unembed:
            processed_dict = ProcessWeights.center_unembed(processed_dict, adapter=adapter)

        if fold_value_biases:
            processed_dict = ProcessWeights.fold_value_biases(processed_dict, cfg, adapter=adapter)

        if refactor_factored_attn_matrices:
            processed_dict = ProcessWeights.refactor_factored_attn_matrices(
                processed_dict, cfg, adapter=adapter
            )

        return processed_dict

    @staticmethod
    def extract_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract state dictionary from an nn.Module, cleaning up any _original_component references.
        
        This function extracts the state dictionary from a PyTorch model and removes any
        _original_component suffixes that might be present in bridge models.
        
        Args:
            model: The PyTorch model to extract state dict from
            
        Returns:
            Dict[str, torch.Tensor]: Cleaned state dictionary without _original_component references
        """
        # If the model has a custom state_dict method (like TransformerBridge), use it directly
        if hasattr(model, 'state_dict') and hasattr(model.__class__, 'state_dict') and model.__class__.state_dict != nn.Module.state_dict:
            return model.state_dict()
        
        # Otherwise, manually clean up _original_component suffixes
        state_dict = model.state_dict()
        cleaned_state_dict = {}
        for key, tensor in state_dict.items():
            clean_key = key.replace("._original_component", "")
            cleaned_state_dict[clean_key] = tensor.clone()
        
        return cleaned_state_dict

    @staticmethod
    def load_processed_weights_into_module(processed_state_dict, module):
        """Load processed weights into an nn.Module.
        
        Args:
            processed_state_dict: Dictionary of processed weights
            module: The nn.Module to load weights into
            
        Returns:
            The same module with processed weights loaded
        """
        import torch

        # If the module has a custom load_state_dict method (like TransformerBridge), use it directly
        if hasattr(module, 'load_state_dict') and hasattr(module.__class__, 'load_state_dict') and module.__class__.load_state_dict != nn.Module.load_state_dict:
            module.load_state_dict(processed_state_dict, strict=False)
            return module
        
        # Otherwise, manually map processed keys to original keys with _original_component suffixes
        original_state_dict = module.state_dict()
        new_state_dict = {}
        
        # Map processed keys to original keys
        for processed_key, processed_tensor in processed_state_dict.items():
            # Find the corresponding key with _original_component suffix
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == processed_key:
                    new_state_dict[orig_key] = processed_tensor
                    # Debug output for QKV weights
                    if 'c_attn.weight' in processed_key:
                        print(f"DEBUG: Mapped {processed_key} -> {orig_key}")
                        print(f"  Processed range: [{processed_tensor.min():.6f}, {processed_tensor.max():.6f}]")
                    break
        
        # Load the new state dict into the module
        module.load_state_dict(new_state_dict, strict=False)
        
        return module
    
    @staticmethod
    def create_model_with_processed_weights(processed_state_dict, original_model, model_class=None):
        """Create a new model instance with processed weights.
        
        Args:
            processed_state_dict: Dictionary of processed weights
            original_model: The original model to use as a template
            model_class: The model class to instantiate (if None, uses type(original_model))
            
        Returns:
            A new model instance with processed weights loaded
        """
        import torch

        # if model_class is None:
        #     model_class = type(original_model)
        # Create a new model instance
        # new_model = model_class(original_model.config)
        # # Get the new model's state dict
        # new_state_dict = new_model.state_dict()
        # # Map processed keys to new model keys
        # for processed_key, processed_tensor in processed_state_dict.items():
        #     # Find the corresponding key in the new model
        #     for new_key in new_state_dict.keys():
        #         if new_key.replace("._original_component", "") == processed_key:
        #             new_state_dict[new_key] = processed_tensor
        #             break
        
        original_model.load_state_dict(processed_state_dict, strict=True, assign=True)
        # print("loading weights")
        # # Load the processed weights into the new model
        # state_dict_keys = list(processed_state_dict.keys())
        # for key in state_dict_keys:
            
        #     del processed_state_dict[key]
        
        return original_model
    
    @staticmethod
    def _get_parameter_by_name(module, param_name):
        """Get a parameter from a module by its name.
        
        Args:
            module: The nn.Module
            param_name: The parameter name (e.g., "transformer.h.0.attn.c_attn.weight")
            
        Returns:
            The parameter tensor or None if not found
        """
        parts = param_name.split('.')
        current = module
        
        try:
            for part in parts:
                current = getattr(current, part)
            return current
        except AttributeError:
            return None
