"""Granular weight processing benchmarks.

This module provides detailed benchmarks that test each weight processing operation
individually and in combination to isolate which processing steps cause issues.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity


@dataclass
class WeightProcessingConfig:
    """Configuration for a specific weight processing test."""

    name: str
    fold_ln: bool
    center_writing_weights: bool
    center_unembed: bool
    fold_value_biases: bool
    refactor_factored_attn_matrices: bool

    def __str__(self) -> str:
        """Get a short string representation."""
        flags = []
        if self.fold_ln:
            flags.append("fold_ln")
        if self.center_writing_weights:
            flags.append("center_weights")
        if self.center_unembed:
            flags.append("center_unembed")
        if self.fold_value_biases:
            flags.append("fold_value_bias")
        if self.refactor_factored_attn_matrices:
            flags.append("refactor_attn")
        return "+".join(flags) if flags else "none"


# Define all weight processing configurations to test
WEIGHT_PROCESSING_CONFIGS = [
    # Individual flags
    WeightProcessingConfig(
        name="only_fold_ln",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="only_center_weights",
        fold_ln=False,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="only_center_unembed",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="only_fold_value_biases",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
    # Common combinations
    WeightProcessingConfig(
        name="fold_ln+center_weights",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+center_weights+center_unembed",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+center_weights+fold_value_biases",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
    # Standard configuration (all enabled except refactor)
    WeightProcessingConfig(
        name="standard_all",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
]


def run_granular_weight_processing_benchmarks(
    model_name: str,
    device: str,
    test_text: str,
    verbose: bool = True,
) -> Dict[str, List[BenchmarkResult]]:
    """Run benchmarks with each weight processing configuration.

    This function tests each weight processing flag individually and in combination
    to identify which specific processing steps cause issues.

    Args:
        model_name: Name of the model to benchmark
        device: Device to run on ("cpu" or "cuda")
        test_text: Test text for generation/inference
        verbose: Whether to print detailed output

    Returns:
        Dictionary mapping config name to list of benchmark results
    """
    from transformer_lens import HookedTransformer
    from transformer_lens.benchmarks.forward_pass import (
        benchmark_logits_equivalence,
        benchmark_loss_equivalence,
    )
    from transformer_lens.benchmarks.hook_registration import (
        benchmark_critical_forward_hooks,
        benchmark_forward_hooks,
        benchmark_hook_functionality,
    )
    from transformer_lens.model_bridge.bridge import TransformerBridge

    all_results: Dict[str, List[BenchmarkResult]] = {}

    if verbose:
        print("\n" + "=" * 80)
        print("GRANULAR WEIGHT PROCESSING BENCHMARKS")
        print(f"Model: {model_name}")
        print(f"Testing {len(WEIGHT_PROCESSING_CONFIGS)} configurations")
        print("=" * 80)

    for config in WEIGHT_PROCESSING_CONFIGS:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {config.name}")
            print(f"Flags: {config}")
            print(f"{'='*80}\n")

        results: List[BenchmarkResult] = []

        try:
            # Load HookedTransformer reference with same processing
            if verbose:
                print(f"Loading HookedTransformer ({config})...")
            ht_ref = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=config.fold_ln,
                center_writing_weights=config.center_writing_weights,
                center_unembed=config.center_unembed,
                fold_value_biases=config.fold_value_biases,
                refactor_factored_attn_matrices=config.refactor_factored_attn_matrices,
            )

            # Load TransformerBridge and apply same processing
            if verbose:
                print(f"Loading TransformerBridge ({config})...")
            bridge = TransformerBridge.boot_transformers(model_name, device=device)
            bridge.enable_compatibility_mode(
                disable_warnings=True,
                fold_ln=config.fold_ln,
                center_writing_weights=config.center_writing_weights,
                center_unembed=config.center_unembed,
            )

            # Run core benchmarks
            if verbose:
                print("Running benchmarks...")

            # Logits/loss equivalence
            results.append(
                benchmark_logits_equivalence(bridge, test_text, reference_model=ht_ref)
            )
            results.append(benchmark_loss_equivalence(bridge, test_text, reference_model=ht_ref))

            # Hook functionality
            results.append(
                benchmark_hook_functionality(bridge, test_text, reference_model=ht_ref)
            )
            results.append(
                benchmark_critical_forward_hooks(bridge, test_text, reference_model=ht_ref)
            )
            results.append(benchmark_forward_hooks(bridge, test_text, reference_model=ht_ref))

            # Clean up
            del bridge
            del ht_ref
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            # Record failure
            results.append(
                BenchmarkResult(
                    name=f"{config.name}_error",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run configuration: {str(e)}",
                    details={"error": str(e), "config": str(config)},
                )
            )

        # Store results
        all_results[config.name] = results

        # Print summary for this config
        if verbose:
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"\n{config.name}: {passed}/{total} passed")

    # Print overall summary
    if verbose:
        print("\n" + "=" * 80)
        print("GRANULAR WEIGHT PROCESSING SUMMARY")
        print("=" * 80)
        for config_name, results in all_results.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            status = "✅" if passed == total else "❌" if passed == 0 else "⚠️"
            print(f"{status} {config_name}: {passed}/{total} passed")

    return all_results
