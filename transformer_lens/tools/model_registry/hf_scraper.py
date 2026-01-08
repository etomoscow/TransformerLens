#!/usr/bin/env python3
"""HuggingFace model scraper for discovering compatible models.

This module queries the HuggingFace Hub API to find ALL models and categorize
them by architecture - those supported by TransformerLens and those not yet supported.

The scraper works by:
1. Scanning ALL text-generation models on HuggingFace (paginated)
2. Extracting the architecture class from each model's config
3. Categorizing models into supported vs unsupported based on TransformerLens adapters
4. Building comprehensive lists for both categories

Usage:
    # Full scan of all HuggingFace models (recommended)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Quick scan (top N models by downloads)
    python -m transformer_lens.tools.model_registry.hf_scraper --limit 10000

    # Output to custom directory
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan --output data/
"""

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional, TypedDict


class GapEntry(TypedDict):
    """Type for architecture gap entries."""

    architecture_id: str
    total_models: int


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Architectures supported by TransformerLens (from architecture_adapter_factory.py)
# These are the exact HuggingFace architecture class names
SUPPORTED_ARCHITECTURE_SET = {
    "BertForMaskedLM",
    "BloomForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "GPT2LMHeadModel",
    "GptOssForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "OPTForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "QwenForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "StableLmForCausalLM",
    "T5ForConditionalGeneration",
}


def get_model_architecture(api, model_id: str) -> Optional[str]:
    """Get the primary architecture class from a model's config.

    Args:
        api: HuggingFace API client
        model_id: The model ID on HuggingFace

    Returns:
        Architecture class name or None if not found
    """
    try:
        info = api.model_info(model_id)
        if info.config and isinstance(info.config, dict):
            archs = info.config.get("architectures", [])
            if archs:
                return archs[0]  # Return primary architecture
    except Exception:
        pass
    return None


def _load_existing_models(output_dir: Path) -> tuple[set[str], list[dict]]:
    """Load model IDs and data already in supported_models.json.

    Args:
        output_dir: Directory containing the data files

    Returns:
        Tuple of (set of existing model IDs, list of existing model dicts)
    """
    existing_ids: set[str] = set()
    existing_models: list[dict] = []
    supported_path = output_dir / "supported_models.json"

    if supported_path.exists():
        try:
            with open(supported_path) as f:
                data = json.load(f)
            existing_models = data.get("models", [])
            for model in existing_models:
                if "model_id" in model:
                    existing_ids.add(model["model_id"])
            logger.info(f"Loaded {len(existing_ids)} existing models from {supported_path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load existing models: {e}")

    return existing_ids, existing_models


def scrape_all_models(
    output_dir: Path,
    max_models: Optional[int] = None,
    task: str = "text-generation",
    batch_size: int = 1000,
    checkpoint_interval: int = 5000,
) -> tuple[dict, dict]:
    """Scrape ALL models from HuggingFace and categorize by architecture.

    This is the comprehensive scraper that:
    1. Loads existing models from supported_models.json to preserve them
    2. Skips models already in the JSON (only scans new models)
    3. Iterates through ALL models for a given task
    4. Fetches the architecture from each model's config
    5. Categorizes into supported vs unsupported
    6. Saves checkpoints periodically for long runs

    Args:
        output_dir: Directory to write JSON data files
        max_models: Maximum NEW models to scan (None = unlimited/all)
        task: HuggingFace task filter (default: text-generation)
        batch_size: Log progress every N models
        checkpoint_interval: Save checkpoint every N models

    Returns:
        Tuple of (supported_models_dict, architecture_gaps_dict)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for scraping. "
            "Install it with: pip install huggingface_hub"
        )

    api = HfApi()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing models from supported_models.json
    existing_model_ids, existing_models = _load_existing_models(output_dir)

    # Track all models by architecture (start with existing models)
    supported_models: list[dict] = list(existing_models)  # Preserve existing
    unsupported_arch_counts: dict[str, int] = {}  # arch -> count (not storing model IDs)

    scanned = 0
    skipped = 0
    new_supported = 0
    errors = 0
    start_time = time.time()

    # Check for existing checkpoint to resume from
    checkpoint_path = output_dir / "scrape_checkpoint.json"
    seen_models: set[str] = set(existing_model_ids)  # Include existing as "seen"

    if checkpoint_path.exists():
        logger.info(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        # Merge checkpoint data with existing
        checkpoint_supported = checkpoint.get("supported_models", [])
        for model in checkpoint_supported:
            if model["model_id"] not in existing_model_ids:
                supported_models.append(model)
                existing_model_ids.add(model["model_id"])
        unsupported_arch_counts = checkpoint.get("unsupported_arch_counts", {})
        seen_models.update(checkpoint.get("seen_models", []))
        scanned = checkpoint.get("scanned", 0)
        skipped = checkpoint.get("skipped", 0)
        logger.info(f"Resumed from checkpoint: {scanned} models already scanned")

    logger.info(f"Starting comprehensive HuggingFace scan for task='{task}'...")
    logger.info(f"Skipping {len(existing_model_ids)} models already in supported_models.json")
    if max_models:
        logger.info(f"Will scan up to {max_models} NEW models")
    else:
        logger.info("Will scan ALL new models (this may take a while)")

    try:
        for model in api.list_models(task=task, sort="downloads", direction=-1):
            # Skip if already in our JSON or processed in this run
            if model.id in seen_models:
                skipped += 1
                continue

            scanned += 1
            seen_models.add(model.id)

            if max_models and scanned > max_models:
                break

            # Get architecture from model config
            arch = get_model_architecture(api, model.id)

            if arch is None:
                errors += 1
            elif arch in SUPPORTED_ARCHITECTURE_SET:
                supported_models.append(
                    {
                        "model_id": model.id,
                        "architecture_id": arch,
                        "verified": False,
                        "verified_at": None,
                        "phase1_score": None,
                        "phase2_score": None,
                        "phase3_score": None,
                    }
                )
                new_supported += 1
            else:
                # Only track count, not individual model IDs
                unsupported_arch_counts[arch] = unsupported_arch_counts.get(arch, 0) + 1

            # Progress logging
            if scanned % batch_size == 0:
                elapsed = time.time() - start_time
                rate = scanned / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Scanned {scanned} new | "
                    f"Skipped {skipped} existing | "
                    f"New supported: {new_supported} | "
                    f"Total supported: {len(supported_models)} | "
                    f"Unsupported archs: {len(unsupported_arch_counts)} | "
                    f"Errors: {errors} | "
                    f"Rate: {rate:.1f}/s"
                )

            # Save checkpoint periodically
            if scanned % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    supported_models,
                    unsupported_arch_counts,
                    list(seen_models),
                    scanned,
                    skipped,
                )
                logger.info(f"Saved checkpoint at {scanned} models")

            # Rate limiting to avoid API issues
            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving checkpoint...")
        _save_checkpoint(
            checkpoint_path,
            supported_models,
            unsupported_arch_counts,
            list(seen_models),
            scanned,
            skipped,
        )
        raise
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        _save_checkpoint(
            checkpoint_path,
            supported_models,
            unsupported_arch_counts,
            list(seen_models),
            scanned,
            skipped,
        )
        raise

    # Build final reports
    elapsed = time.time() - start_time
    logger.info(f"\nScan complete in {elapsed:.1f}s")
    logger.info(f"New models scanned: {scanned}")
    logger.info(f"Existing models skipped: {skipped}")
    logger.info(f"New supported models found: {new_supported}")
    logger.info(f"Total supported models: {len(supported_models)}")
    logger.info(f"Unsupported architectures found: {len(unsupported_arch_counts)}")

    # Build supported models report
    supported_report = {
        "generated_at": date.today().isoformat(),
        "scan_info": {
            "total_scanned": scanned,
            "task_filter": task,
            "scan_duration_seconds": round(elapsed, 1),
        },
        "models": supported_models,
    }

    # Write supported models
    supported_path = output_dir / "supported_models.json"
    with open(supported_path, "w") as f:
        json.dump(supported_report, f, indent=2)
    logger.info(f"Wrote {len(supported_models)} supported models to {supported_path}")

    # Build architecture gaps report (counts only, no model IDs)
    gaps: list[GapEntry] = [
        {
            "architecture_id": arch,
            "total_models": count,
        }
        for arch, count in sorted(unsupported_arch_counts.items(), key=lambda x: -x[1])
    ]

    gaps_report = {
        "generated_at": date.today().isoformat(),
        "scan_info": {
            "total_scanned": scanned,
            "task_filter": task,
        },
        "total_unsupported_architectures": len(gaps),
        "total_unsupported_models": sum(unsupported_arch_counts.values()),
        "gaps": gaps,
    }

    gaps_path = output_dir / "architecture_gaps.json"
    with open(gaps_path, "w") as f:
        json.dump(gaps_report, f, indent=2)
    logger.info(f"Wrote {len(gaps)} architecture gaps to {gaps_path}")

    # Write verification history placeholder
    verification_path = output_dir / "verification_history.json"
    if not verification_path.exists():
        with open(verification_path, "w") as f:
            json.dump(
                {
                    "last_updated": datetime.now().isoformat(),
                    "records": [],
                },
                f,
                indent=2,
            )

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file (scan complete)")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total models scanned: {scanned}")
    logger.info(f"\nSUPPORTED ARCHITECTURES ({len(SUPPORTED_ARCHITECTURE_SET)}):")

    # Count models per supported architecture
    supported_arch_counts: dict[str, int] = {}
    for model in supported_models:
        arch = model["architecture_id"]
        supported_arch_counts[arch] = supported_arch_counts.get(arch, 0) + 1

    for arch, count in sorted(supported_arch_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {arch}: {count} models")

    logger.info(f"\nTOP 20 UNSUPPORTED ARCHITECTURES (of {len(gaps)}):")
    for gap in gaps[:20]:
        logger.info(f"  {gap['architecture_id']}: {gap['total_models']} models")

    if len(gaps) > 20:
        remaining = sum(g["total_models"] for g in gaps[20:])
        logger.info(f"  ... and {len(gaps) - 20} more architectures ({remaining} models)")

    logger.info("=" * 70)

    return supported_report, gaps_report


def _save_checkpoint(
    path: Path,
    supported_models: list,
    unsupported_arch_counts: dict,
    seen_models: list,
    scanned: int,
    skipped: int = 0,
):
    """Save scraping progress to a checkpoint file."""
    checkpoint = {
        "supported_models": supported_models,
        "unsupported_arch_counts": unsupported_arch_counts,
        "seen_models": seen_models,
        "scanned": scanned,
        "skipped": skipped,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape HuggingFace to find all TransformerLens-compatible models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full scan of ALL text-generation models (recommended)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Quick scan of top 10,000 models by downloads
    python -m transformer_lens.tools.model_registry.hf_scraper --limit 10000

    # Resume interrupted scan (checkpoints are saved automatically)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Output to custom directory
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan -o ./my_data/
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for JSON data files (default: ./data/)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan ALL models on HuggingFace (may take hours, saves checkpoints)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum models to scan (default: 10000, ignored with --full-scan)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="HuggingFace task to filter by (default: text-generation)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N models (default: 5000)",
    )

    args = parser.parse_args()

    max_models = None if args.full_scan else args.limit

    scrape_all_models(
        output_dir=args.output,
        max_models=max_models,
        task=args.task,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
