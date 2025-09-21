#!/usr/bin/env python3
"""
Unit tests for TransformerBridge state_dict functionality.

This module tests that the bridge properly handles state_dict operations by:
1. Filtering out _original_component references when getting state_dict
2. Mapping clean keys back to _original_component keys when loading state_dict
3. Supporting both original model keys and TransformerLens keys
"""


import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge


class MockAdapter(ArchitectureAdapter):
    """Mock adapter for testing."""

    def __init__(self):
        self.cfg = type(
            "Config", (), {"n_layers": 1, "d_model": 10, "n_heads": 2, "d_head": 5, "device": "cpu"}
        )()
        self.component_mapping = {}

    def get_component_mapping(self):
        return {}

    def get_remote_component(self, model, path):
        return getattr(model, path)

    def translate_transformer_lens_path(self, path):
        return path


class MockModelWithOriginalComponent(nn.Module):
    """Test model that simulates having _original_component references."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.embedding = nn.Embedding(100, 10)
        # Simulate the bridge adding _original_component references
        self.add_module("_original_component", nn.Linear(10, 5))

    def __getattr__(self, name):
        """Allow access to OV and other components through _original_component."""
        if name == "OV":
            # Return a mock OV component
            return nn.Linear(5, 10)
        return super().__getattr__(name)


class MockTransformerBridge(TransformerBridge):
    """Test bridge that doesn't require full initialization."""

    def __init__(self, original_model):
        # Skip the full initialization to avoid hook setup issues
        nn.Module.__init__(self)
        self.original_model = original_model
        self.adapter = MockAdapter()
        self.cfg = self.adapter.cfg
        self.tokenizer = None
        self.compatibility_mode = False
        self._hook_cache = None
        self._hook_registry = {}


class TestBridgeStateDict:
    """Test cases for bridge state_dict functionality."""

    def test_state_dict_filters_original_component_references(self):
        """Test that state_dict() filters out _original_component references but preserves submodules."""
        # Create test model with _original_component references
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get state dict
        state_dict = bridge.state_dict()

        # Verify no direct _original_component references in the keys
        has_original_component = any("_original_component" in key for key in state_dict.keys())
        assert (
            not has_original_component
        ), f"Found _original_component references: {[k for k in state_dict.keys() if '_original_component' in k]}"

        # Verify we have the expected clean keys
        expected_keys = {"linear.weight", "linear.bias", "embedding.weight"}
        actual_keys = set(state_dict.keys())
        assert actual_keys == expected_keys, f"Expected {expected_keys}, got {actual_keys}"

        # Verify that submodules like OV are still accessible via __getattr__
        # This tests that the _original_component module itself is filtered but its submodules are accessible
        try:
            ov_component = bridge.OV
            assert ov_component is not None, "OV component should be accessible"
        except AttributeError:
            # This is expected if the mock model doesn't have an OV component
            pass

    def test_load_state_dict_with_clean_keys(self):
        """Test that load_state_dict() accepts clean keys and maps them correctly."""
        # Create test model with _original_component references
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get initial state dict
        initial_state_dict = bridge.state_dict()

        # Create modified state dict with clean keys
        modified_state_dict = {}
        for key, tensor in initial_state_dict.items():
            modified_state_dict[key] = tensor + 0.1

        # Load the modified state dict
        missing_keys, unexpected_keys = bridge.load_state_dict(modified_state_dict, strict=False)

        # Verify no unexpected keys (missing keys are expected for _original_component)
        assert len(unexpected_keys) == 0, f"Unexpected unexpected keys: {unexpected_keys}"
        # Missing keys should only be _original_component keys
        expected_missing = {"_original_component.weight", "_original_component.bias"}
        actual_missing = set(missing_keys)
        assert (
            actual_missing == expected_missing
        ), f"Expected missing keys {expected_missing}, got {actual_missing}"

        # Verify weights were actually loaded
        new_state_dict = bridge.state_dict()
        for key in initial_state_dict.keys():
            expected_weight = modified_state_dict[key]  # This is original + 0.1
            new_weight = new_state_dict[key]
            assert torch.allclose(
                new_weight, expected_weight, atol=1e-6
            ), f"Weight for {key} was not loaded correctly"

    def test_load_state_dict_with_original_component_keys(self):
        """Test that load_state_dict() accepts keys with _original_component references."""
        # Create test model with _original_component references
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get the raw state dict (with _original_component references)
        raw_state_dict = test_model.state_dict()

        # Create modified state dict with ALL keys (including _original_component keys)
        modified_state_dict = {}
        for key, tensor in raw_state_dict.items():
            modified_state_dict[key] = tensor + 0.2

        # Load the modified state dict
        missing_keys, unexpected_keys = bridge.load_state_dict(modified_state_dict, strict=False)

        # Verify no missing or unexpected keys when loading with original keys
        assert len(missing_keys) == 0, f"Unexpected missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected unexpected keys: {unexpected_keys}"

        # Verify weights were loaded correctly
        new_state_dict = bridge.state_dict()
        for key in modified_state_dict.keys():
            if not key.startswith("_original_component") and key in new_state_dict:
                expected_weight = modified_state_dict[key]  # This is original + 0.2
                new_weight = new_state_dict[key]
                assert torch.allclose(
                    new_weight, expected_weight, atol=1e-6
                ), f"Weight for {key} was not loaded correctly"

    def test_round_trip_state_dict_operations(self):
        """Test round-trip: save -> modify -> load -> save."""
        # Create test model
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get initial weights
        initial_weights = bridge.state_dict()

        # Modify weights
        modified_weights = {k: v + 0.3 for k, v in initial_weights.items()}

        # Load modified weights
        bridge.load_state_dict(modified_weights, strict=False)

        # Verify weights were loaded
        final_weights = bridge.state_dict()
        for key in initial_weights.keys():
            expected_weight = modified_weights[key]  # This is initial + 0.3
            actual_weight = final_weights[key]
            assert torch.allclose(
                expected_weight, actual_weight, atol=1e-6
            ), f"Round-trip failed for {key}"

    def test_state_dict_with_transformer_lens_keys(self):
        """Test state_dict operations with TransformerLens-style keys."""

        # Create a simple model with TL-style structure
        class MockTLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 10)
                self.unembed = nn.Linear(10, 100)
                self.linear = nn.Linear(10, 10)

        # Create bridge with mock model
        mock_model = MockTLModel()
        bridge = MockTransformerBridge(mock_model)

        # Test state_dict with TL keys
        state_dict = bridge.state_dict()

        # Verify we get clean keys (no _original_component references)
        has_original_component = any("_original_component" in key for key in state_dict.keys())
        assert (
            not has_original_component
        ), f"Found _original_component references: {[k for k in state_dict.keys() if '_original_component' in k]}"

        # Test loading with TL keys
        modified_state_dict = {k: v + 0.1 for k, v in state_dict.items()}
        missing_keys, unexpected_keys = bridge.load_state_dict(modified_state_dict, strict=False)

        # Verify successful loading
        assert len(missing_keys) == 0, f"Unexpected missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected unexpected keys: {unexpected_keys}"

    def test_state_dict_preserves_tensor_properties(self):
        """Test that state_dict operations preserve tensor properties (device, dtype, etc.)."""
        # Create test model on CPU
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get state dict
        state_dict = bridge.state_dict()

        # Verify tensor properties are preserved
        for key, tensor in state_dict.items():
            assert isinstance(tensor, torch.Tensor), f"{key} is not a tensor"
            assert tensor.device.type == "cpu", f"{key} is not on CPU"
            assert tensor.dtype == torch.float32, f"{key} is not float32"

    def test_state_dict_with_different_prefixes(self):
        """Test state_dict operations with different prefix scenarios."""
        # Create test model
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Test with prefix
        state_dict_with_prefix = bridge.state_dict(prefix="model.")

        # Verify prefix is applied
        for key in state_dict_with_prefix.keys():
            assert key.startswith("model."), f"Key {key} does not have prefix 'model.'"

        # Test loading with prefix
        modified_with_prefix = {k: v + 0.1 for k, v in state_dict_with_prefix.items()}
        missing_keys, unexpected_keys = bridge.load_state_dict(modified_with_prefix, strict=False)

        # Should have missing keys because we're loading with prefix but model expects no prefix
        assert len(missing_keys) > 0, "Expected missing keys when loading with prefix"

    def test_state_dict_strict_mode(self):
        """Test state_dict loading in strict mode."""
        # Create test model
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get initial state dict
        initial_state_dict = bridge.state_dict()

        # Create state dict with extra keys
        extra_state_dict = initial_state_dict.copy()
        extra_state_dict["nonexistent.weight"] = torch.randn(5, 10)

        # Test strict mode (should fail with unexpected keys)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            bridge.load_state_dict(extra_state_dict, strict=True)

        # Test non-strict mode (should succeed)
        missing_keys, unexpected_keys = bridge.load_state_dict(extra_state_dict, strict=False)
        assert "nonexistent.weight" in unexpected_keys, "Extra key should be in unexpected_keys"

    def test_load_state_dict_with_mixed_keys(self):
        """Test that load_state_dict() accepts a mix of clean keys and original keys."""
        # Create test model with _original_component references
        test_model = MockModelWithOriginalComponent()
        bridge = MockTransformerBridge(test_model)

        # Get both clean and raw state dicts
        clean_state_dict = bridge.state_dict()
        raw_state_dict = test_model.state_dict()

        # Create a mixed state dict with both clean and original keys
        mixed_state_dict = {}
        # Add some clean keys
        for key, tensor in clean_state_dict.items():
            if key == "linear.weight":  # Only add one clean key
                mixed_state_dict[key] = tensor + 0.3
                break

        # Add some original keys (including _original_component)
        for key, tensor in raw_state_dict.items():
            if key == "linear.bias":  # Add one original key
                mixed_state_dict[key] = tensor + 0.3
                break

        # Load the mixed state dict
        missing_keys, unexpected_keys = bridge.load_state_dict(mixed_state_dict, strict=False)

        # Should have missing keys for the keys we didn't include
        assert len(unexpected_keys) == 0, f"Unexpected unexpected keys: {unexpected_keys}"

        # Verify the weights we loaded were loaded correctly
        new_state_dict = bridge.state_dict()
        for key in mixed_state_dict.keys():
            if not key.startswith("_original_component") and key in new_state_dict:
                expected_weight = mixed_state_dict[key]
                new_weight = new_state_dict[key]
                assert torch.allclose(
                    new_weight, expected_weight, atol=1e-6
                ), f"Weight for {key} was not loaded correctly"

    def test_state_dict_filtering_preserves_submodules(self):
        """Test that state_dict filtering preserves submodules while filtering _original_component."""
        # Create a simple test model
        test_model = nn.Module()
        test_model.linear = nn.Linear(10, 5)
        test_model.embedding = nn.Embedding(100, 10)
        # Add a mock OV component directly
        test_model.OV = nn.Linear(5, 10)
        # Simulate the bridge adding _original_component references
        test_model.add_module("_original_component", nn.Linear(10, 5))

        bridge = MockTransformerBridge(test_model)

        # Get state dict (this should filter out _original_component but preserve submodules)
        state_dict = bridge.state_dict()

        # Verify no _original_component references in state_dict
        has_original_component = any("_original_component" in key for key in state_dict.keys())
        assert (
            not has_original_component
        ), f"Found _original_component references: {[k for k in state_dict.keys() if '_original_component' in k]}"

        # Verify that submodules like OV are still in the state_dict
        ov_keys = [k for k in state_dict.keys() if "OV" in k]
        assert len(ov_keys) > 0, "OV component should be present in state_dict"

        # Verify we have the expected clean keys
        expected_keys = {"linear.weight", "linear.bias", "embedding.weight", "OV.weight", "OV.bias"}
        actual_keys = set(state_dict.keys())
        assert actual_keys == expected_keys, f"Expected {expected_keys}, got {actual_keys}"

    def test_state_dict_with_empty_model(self):
        """Test state_dict operations with an empty model."""
        # Create empty model
        empty_model = nn.Module()
        bridge = MockTransformerBridge(empty_model)

        # Test state_dict
        state_dict = bridge.state_dict()
        assert len(state_dict) == 0, "Empty model should have empty state_dict"

        # Test load_state_dict
        missing_keys, unexpected_keys = bridge.load_state_dict({}, strict=False)
        assert len(missing_keys) == 0, "Should have no missing keys for empty model"
        assert len(unexpected_keys) == 0, "Should have no unexpected keys for empty model"


if __name__ == "__main__":
    pytest.main([__file__])
