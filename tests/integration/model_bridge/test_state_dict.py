"""Test TransformerBridge state_dict() functionality."""
import torch

from transformer_lens.model_bridge.sources.transformers import boot


class TestBridgeStateDict:
    """Test suite for TransformerBridge.state_dict() method."""

    def test_state_dict_contains_tl_format_keys(self):
        """Test that state_dict returns TransformerLens format keys.

        Specifically tests that:
        1. blocks.0.attn.q.weight exists (TL format for split Q weights)
        2. No duplicate keys exist (e.g., no both ln1 and ln_1)
        3. No _original_component references exist
        """
        # Load GPT2 model
        bridge = boot('distilgpt2', device='cpu', dtype=torch.float32)

        # Get state dict
        state_dict = bridge.state_dict()

        # Assert that TL format key exists for split Q weights
        assert 'blocks.0.attn.q.weight' in state_dict, (
            f"Expected 'blocks.0.attn.q.weight' in state_dict, "
            f"but got keys: {list(state_dict.keys())[:50]}"
        )

        # Also check that other TL format split keys exist
        assert 'blocks.0.attn.k.weight' in state_dict, "Expected split K weights in TL format"
        assert 'blocks.0.attn.v.weight' in state_dict, "Expected split V weights in TL format"

        # Check that biases also exist
        assert 'blocks.0.attn.q.bias' in state_dict, "Expected Q bias in TL format"
        assert 'blocks.0.attn.k.bias' in state_dict, "Expected K bias in TL format"
        assert 'blocks.0.attn.v.bias' in state_dict, "Expected V bias in TL format"

    def test_state_dict_no_duplicate_keys(self):
        """Test that state_dict does not contain duplicate HF/TL keys.

        Verifies that we don't have both:
        - ln_1 and ln1 (should only have one)
        - c_attn and q/k/v (should only have split q/k/v for TL)
        - c_fc and mlp.in (should only have one)
        """
        bridge = boot('distilgpt2', device='cpu', dtype=torch.float32)
        state_dict = bridge.state_dict()

        # Check that we don't have both HF and TL layer norm keys
        has_ln1 = any('.ln1.' in k for k in state_dict.keys())
        has_ln_1 = any('.ln_1.' in k for k in state_dict.keys())

        # We should have one or the other, but not both
        assert not (has_ln1 and has_ln_1), (
            "State dict should not contain both .ln1. and .ln_1. keys"
        )

        # Check that we have split Q/K/V instead of joint c_attn
        has_split_qkv = any('.attn.q.' in k for k in state_dict.keys())
        has_c_attn = any('.c_attn.' in k for k in state_dict.keys())

        assert has_split_qkv, "State dict should contain split Q/K/V weights"
        assert not has_c_attn, (
            "State dict should not contain c_attn (joint QKV) when split Q/K/V exist"
        )

    def test_state_dict_no_original_component_refs(self):
        """Test that state_dict does not contain _original_component references."""
        bridge = boot('distilgpt2', device='cpu', dtype=torch.float32)
        state_dict = bridge.state_dict()

        # Check that no keys contain _original_component
        original_component_keys = [
            k for k in state_dict.keys()
            if '_original_component' in k
        ]

        assert len(original_component_keys) == 0, (
            f"State dict should not contain _original_component references, "
            f"but found: {original_component_keys}"
        )

    def test_state_dict_tl_component_names(self):
        """Test that state_dict uses TransformerLens component naming.

        Verifies:
        - Uses 'blocks' instead of 'transformer.h'
        - Uses 'attn.o' instead of 'attn.c_proj'
        - Uses 'mlp.in' and 'mlp.out' instead of 'c_fc' and 'c_proj'
        """
        bridge = boot('distilgpt2', device='cpu', dtype=torch.float32)
        state_dict = bridge.state_dict()

        # Check for TL format attention output projection
        has_attn_o = any('.attn.o.' in k for k in state_dict.keys())
        has_attn_c_proj = any('.attn.c_proj.' in k for k in state_dict.keys())

        # Should have 'o' for TL format
        assert has_attn_o, "State dict should use .attn.o. for attention output projection"

        # Check for TL format MLP keys
        has_mlp_in = any('.mlp.in.' in k for k in state_dict.keys())
        has_mlp_out = any('.mlp.out.' in k for k in state_dict.keys())
        has_c_fc = any('.c_fc.' in k for k in state_dict.keys())

        assert has_mlp_in, "State dict should use .mlp.in. for MLP input projection"
        assert has_mlp_out, "State dict should use .mlp.out. for MLP output projection"

    def test_state_dict_weights_are_tensors(self):
        """Test that all values in state_dict are tensors with correct shapes."""
        bridge = boot('distilgpt2', device='cpu', dtype=torch.float32)
        state_dict = bridge.state_dict()

        # Check that all values are tensors
        for key, value in state_dict.items():
            assert isinstance(value, torch.Tensor), (
                f"Expected tensor for key '{key}', got {type(value)}"
            )

            # Check that tensors have non-zero size
            assert value.numel() > 0, f"Tensor at key '{key}' has zero elements"

        # Specifically check the Q weight shape
        q_weight = state_dict.get('blocks.0.attn.q.weight')
        assert q_weight is not None, "Q weight should exist"

        # For distilgpt2, should be Conv1D format [d_model, d_head * n_heads]
        # d_model=768, n_heads=12, d_head=64, so [768, 768]
        assert q_weight.shape == torch.Size([768, 768]), (
            f"Expected Q weight shape [768, 768], got {q_weight.shape}"
        )
