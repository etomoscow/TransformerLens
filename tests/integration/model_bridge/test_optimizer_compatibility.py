"""Integration test for TransformerBridge optimizer compatibility.

Tests that TransformerBridge works correctly with PyTorch optimizers,
including parameter access, gradient flow, and parameter updates.
"""

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


def test_optimizer_workflow():
    """Test complete optimizer workflow with TransformerBridge."""
    # Load model
    bridge = TransformerBridge.boot_transformers("distilgpt2")

    # Verify parameters() returns leaf tensors
    params = list(bridge.parameters())
    assert len(params) > 0, "Should have parameters"
    assert all(p.is_leaf for p in params), "All parameters should be leaf tensors"

    # Verify optimizer creation succeeds
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    assert optimizer is not None, "Optimizer should be created successfully"

    # Verify tl_parameters() returns TL-style dict
    tl_params = bridge.tl_parameters()
    assert len(tl_params) > 0, "Should have TL-style parameters"
    assert any(
        "blocks." in name and ".attn." in name for name in tl_params.keys()
    ), "Should have TL-style parameter names like 'blocks.0.attn.W_Q'"

    # Verify tl_named_parameters() iterator matches dict
    tl_named_params = list(bridge.tl_named_parameters())
    assert len(tl_named_params) == len(
        tl_params
    ), "Iterator should yield same number of parameters as dict"
    iterator_dict = dict(tl_named_params)
    for name, tensor in tl_params.items():
        assert name in iterator_dict, f"Name {name} should be in iterator output"
        assert torch.equal(iterator_dict[name], tensor), f"Tensor for {name} should match"

    # Verify named_parameters() returns HF-style names
    hf_names = [name for name, _ in bridge.named_parameters()]
    assert len(hf_names) > 0, "Should have HF-style parameters"
    assert any(
        "_original_component" in name for name in hf_names
    ), "Should have HuggingFace-style parameter names"

    # Verify forward pass and backward work
    device = next(bridge.parameters()).device
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device=device)
    logits = bridge(input_ids)
    expected_shape = (1, 10, bridge.cfg.d_vocab)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"

    loss = logits[0, -1].sum()
    loss.backward()

    # Verify gradients were computed
    params_with_grad = [p for p in bridge.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, "Should have parameters with gradients after backward()"

    # Verify optimizer step updates parameters
    param_before = list(bridge.parameters())[0].clone()
    optimizer.step()
    param_after = list(bridge.parameters())[0]
    assert not torch.allclose(
        param_before, param_after
    ), "Parameters should be updated after optimizer.step()"


def test_optimizer_compatibility_after_compatibility_mode():
    """Test that optimizer still works after enabling compatibility mode."""
    bridge = TransformerBridge.boot_transformers("distilgpt2")
    bridge.enable_compatibility_mode(no_processing=True)

    # Verify parameters are still leaf tensors after compatibility mode
    params = list(bridge.parameters())
    assert all(
        p.is_leaf for p in params
    ), "All parameters should still be leaf tensors after compatibility mode"

    # Verify optimizer works after compatibility mode
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    device = next(bridge.parameters()).device
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device=device)

    logits = bridge(input_ids)
    loss = logits[0, -1].sum()
    loss.backward()
    optimizer.step()

    # If we got here without errors, the test passed
    assert True, "Optimizer should work after compatibility mode"
