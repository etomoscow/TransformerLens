#!/usr/bin/env python3
"""
Integration Test for Weight Conversion Compatibility
==================================================

This test verifies that all weights and biases in HookedTransformer are equal to all weights 
and biases in TransformerBridge after they have been converted by the ProcessWeights conversion 
function. This ensures that the weight processing pipeline maintains mathematical equivalence 
between the two model implementations.

The test:
1. Instantiates both HookedTransformer and TransformerBridge models
2. Tests conversion in both directions (TL->HF and HF->TL) and bidirectional conversion (TL->HF->TL)
3. Compares all weights and biases between the two models
4. Tests multiple model architectures
"""

from typing import Tuple

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.weight_processing import ProcessWeights

# Test models - using smaller models for faster testing
TEST_MODELS = [
    "gpt2",  # GPT-2 architecture
    "google/gemma-2-2b-it",  # Gemma-2 architecture
    "EleutherAI/gpt-neo-125M",  # GPT-Neo architecture
    "EleutherAI/pythia-70m",  # Pythia architecture
]

# Skip models that might not be available in all environments
SKIP_MODELS = []


class TestWeightConversionCompatibility:
    """Test class for weight conversion compatibility between HookedTransformer and TransformerBridge."""

    @pytest.fixture(scope="class", params=TEST_MODELS)
    def model_name(self, request):
        """Model name to use for testing."""
        model = request.param
        if model in SKIP_MODELS:
            pytest.skip(f"Skipping {model} as it's in the skip list")
        return model

    @pytest.fixture(scope="class")
    def device(self):
        """Device to use for testing."""
        return "cpu"

    @pytest.fixture(scope="class")
    def tolerance(self):
        """Numerical tolerance for weight comparisons."""
        return 1e-6

    @pytest.fixture(scope="class")
    def hooked_transformer(self, model_name, device):
        """Load HookedTransformer without processing."""
        print(f"Loading HookedTransformer without processing for {model_name}...")
        try:
            return HookedTransformer.from_pretrained_no_processing(model_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to load HookedTransformer for {model_name}: {e}")

    @pytest.fixture(scope="class")
    def transformer_bridge(self, model_name, device):
        """Load TransformerBridge without processing."""
        print(f"Loading TransformerBridge without processing for {model_name}...")
        try:
            bridge = TransformerBridge.boot_transformers(model_name, device=device)
            bridge.enable_compatibility_mode(no_processing=True)
            return bridge
        except Exception as e:
            pytest.skip(f"Failed to load TransformerBridge for {model_name}: {e}")

    def compare_weight_tensors(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor, key: str, tolerance: float
    ) -> Tuple[bool, str]:
        """Compare two weight tensors and return whether they match.

        Args:
            tensor1: First tensor to compare
            tensor2: Second tensor to compare
            key: Parameter name for error reporting
            tolerance: Numerical tolerance for comparison

        Returns:
            Tuple of (matches, error_message)
        """
        # Check shapes match
        if tensor1.shape != tensor2.shape:
            return False, f"Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}"

        # Check dtypes match
        if tensor1.dtype != tensor2.dtype:
            return False, f"Dtype mismatch for {key}: {tensor1.dtype} vs {tensor2.dtype}"

        # Check if tensors are close
        if not torch.allclose(tensor1, tensor2, atol=tolerance, rtol=tolerance):
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            return False, f"Values don't match for {key}: max difference = {max_diff:.2e}"

        return True, ""

    def test_hf_to_tl_conversion_compatibility(
        self, hooked_transformer, transformer_bridge, tolerance, model_name
    ):
        """Test that HF->TL conversion maintains weight equivalence."""

        print(f"\n=== Testing HF->TL conversion for {model_name} ===")

        # Get weights from both models
        hooked_tensors = hooked_transformer.state_dict()
        bridge_tensors = transformer_bridge.state_dict()

        hooked_tensor_names = set(hooked_tensors.keys())
        bridge_tensor_names = set(bridge_tensors.keys())

        print(f"HookedTransformer has {len(hooked_tensor_names)} parameters")
        print(f"TransformerBridge has {len(bridge_tensor_names)} parameters")

        # Compare all common parameters
        mismatched_params = []
        total_params = len(hooked_tensor_names)
        matched_params = 0

        for key in sorted(hooked_tensor_names):
            hooked_tensor = hooked_tensors[key]

            # Convert bridge tensor from HF format to TL format
            try:
                bridge_tensor = ProcessWeights.convert_tensor_to_tl_format(
                    key, transformer_bridge.adapter, bridge_tensors, transformer_bridge.cfg
                )
            except Exception as e:
                print(f"Failed to convert {key}: {e}")
                mismatched_params.append((key, f"Conversion failed: {e}"))
                continue

            matches, error_msg = self.compare_weight_tensors(
                hooked_tensor, bridge_tensor, key, tolerance
            )

            if matches:
                matched_params += 1
            else:
                mismatched_params.append((key, error_msg))
                print(f"MISMATCH: {error_msg}")

        # Print summary
        print(f"\nHF->TL conversion summary:")
        print(f"Total common parameters: {total_params}")
        print(f"Matched parameters: {matched_params}")
        print(f"Mismatched parameters: {len(mismatched_params)}")

        # Assertions
        if mismatched_params:
            print(f"\nMismatched parameters:")
            for key, error in mismatched_params[:10]:  # Show first 10
                print(f"  - {key}: {error}")

        # The test should pass if all common parameters match
        assert len(mismatched_params) == 0, (
            f"Found {len(mismatched_params)} mismatched parameters out of {total_params} common parameters. "
            f"This indicates that HF->TL conversion is not maintaining equivalence between "
            f"HookedTransformer and TransformerBridge."
        )

        # Additional check: ensure we have a reasonable number of common parameters
        assert total_params > 0, "No common parameters found between models"
        assert total_params >= 10, f"Too few common parameters ({total_params})"

    def test_tl_to_hf_conversion_compatibility(
        self, hooked_transformer, transformer_bridge, tolerance, model_name
    ):
        """Test that TL->HF conversion maintains weight equivalence."""

        print(f"\n=== Testing TL->HF conversion for {model_name} ===")

        # Get weights from both models
        hooked_tensors = hooked_transformer.state_dict()
        bridge_tensors = transformer_bridge.state_dict()

        hooked_tensor_names = set(hooked_tensors.keys())
        bridge_tensor_names = set(bridge_tensors.keys())

        print(f"Testing TL->HF conversion for {len(hooked_tensor_names)} parameters")

        # Compare all common parameters
        mismatched_params = []
        total_params = len(hooked_tensor_names)
        matched_params = 0

        for key in sorted(hooked_tensor_names):
            hooked_tensor = hooked_tensors[key]
            bridge_tensor = getattr(transformer_bridge, key)

            # Convert hooked tensor from TL format to HF format
            try:
                converted_hooked_tensor = ProcessWeights.convert_tensor_to_hf_format(
                    hooked_tensor, key, transformer_bridge.adapter, transformer_bridge.cfg
                )
            except Exception as e:
                print(f"Failed to convert {key}: {e}")
                mismatched_params.append((key, f"Conversion failed: {e}"))
                continue

            matches, error_msg = self.compare_weight_tensors(
                converted_hooked_tensor, bridge_tensor, key, tolerance
            )

            if matches:
                matched_params += 1
            else:
                mismatched_params.append((key, error_msg))
                print(f"MISMATCH: {error_msg}")

        # Print summary
        print(f"\nTL->HF conversion summary:")
        print(f"Total common parameters: {total_params}")
        print(f"Matched parameters: {matched_params}")
        print(f"Mismatched parameters: {len(mismatched_params)}")

        # Assertions
        if mismatched_params:
            print(f"\nMismatched parameters:")
            for key, error in mismatched_params[:10]:  # Show first 10
                print(f"  - {key}: {error}")

        # The test should pass if all common parameters match
        assert len(mismatched_params) == 0, (
            f"Found {len(mismatched_params)} mismatched parameters out of {total_params} common parameters. "
            f"This indicates that TL->HF conversion is not maintaining equivalence between "
            f"HookedTransformer and TransformerBridge."
        )

        # Additional check: ensure we have a reasonable number of common parameters
        assert total_params > 0, "No common parameters found between models"
        assert total_params >= 10, f"Too few common parameters ({total_params})"

    def test_bidirectional_conversion_consistency(
        self, hooked_transformer, transformer_bridge, tolerance, model_name
    ):
        """Test that bidirectional conversion (TL->HF->TL) maintains consistency."""

        print(f"\n=== Testing bidirectional conversion consistency for {model_name} ===")

        # Get weights from both models
        hooked_tensors = hooked_transformer.state_dict()
        bridge_tensors = transformer_bridge.state_dict()

        hooked_tensor_names = set(hooked_tensors.keys())
        bridge_tensor_names = set(bridge_tensors.keys())

        print(f"Testing bidirectional conversion for {len(hooked_tensor_names)} parameters")

        # Test bidirectional conversion
        mismatched_params = []
        total_params = len(hooked_tensor_names)
        matched_params = 0

        for key in sorted(hooked_tensor_names):
            original_tensor = hooked_tensors[key]

            try:
                # Convert TL->HF->TL
                hf_tensor = ProcessWeights.convert_tensor_to_hf_format(
                    original_tensor, key, transformer_bridge.adapter, transformer_bridge.cfg
                )
                back_to_tl_tensor = ProcessWeights.convert_tensor_to_tl_format(
                    key,
                    transformer_bridge.adapter,
                    {transformer_bridge.adapter.translate_transformer_lens_path(key): hf_tensor},
                    transformer_bridge.cfg,
                )
            except Exception as e:
                print(f"Failed bidirectional conversion for {key}: {e}")
                mismatched_params.append((key, f"Bidirectional conversion failed: {e}"))
                continue

            matches, error_msg = self.compare_weight_tensors(
                original_tensor, back_to_tl_tensor, key, tolerance
            )

            if matches:
                matched_params += 1
            else:
                mismatched_params.append((key, error_msg))
                print(f"MISMATCH: {error_msg}")

        # Print summary
        print(f"\nBidirectional conversion summary:")
        print(f"Total common parameters: {total_params}")
        print(f"Matched parameters: {matched_params}")
        print(f"Mismatched parameters: {len(mismatched_params)}")

        # Assertions
        if mismatched_params:
            print(f"\nMismatched parameters:")
            for key, error in mismatched_params[:10]:  # Show first 10
                print(f"  - {key}: {error}")

        # The test should pass if all common parameters match
        assert len(mismatched_params) == 0, (
            f"Found {len(mismatched_params)} mismatched parameters out of {total_params} common parameters. "
            f"This indicates that bidirectional conversion is not maintaining consistency."
        )

        # Additional check: ensure we have a reasonable number of common parameters
        assert total_params > 0, "No common parameters found between models"
        assert total_params >= 10, f"Too few common parameters ({total_params})"
