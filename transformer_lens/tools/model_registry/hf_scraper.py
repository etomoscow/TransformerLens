#!/usr/bin/env python3
"""HuggingFace model scraper for discovering compatible models.

This module queries the HuggingFace Hub API to find all models that use
architectures supported by TransformerLens.

Usage:
    python -m transformer_lens.tools.model_registry.hf_scraper
    python -m transformer_lens.tools.model_registry.hf_scraper --output data/
"""

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Map architecture class names to their HuggingFace tags
# These tags appear in model.tags on HuggingFace
ARCHITECTURE_TO_TAGS = {
    "GPT2LMHeadModel": ["gpt2"],
    "GPTNeoForCausalLM": ["gpt_neo", "gpt-neo"],
    "GPTNeoXForCausalLM": ["gpt_neox", "gpt-neox", "pythia"],
    "GPTJForCausalLM": ["gptj", "gpt-j"],
    "LlamaForCausalLM": ["llama"],
    "MistralForCausalLM": ["mistral"],
    "MixtralForCausalLM": ["mixtral"],
    "GemmaForCausalLM": ["gemma"],
    "Gemma2ForCausalLM": ["gemma2"],
    "Qwen2ForCausalLM": ["qwen2", "qwen"],
    "BloomForCausalLM": ["bloom"],
    "OPTForCausalLM": ["opt"],
    "PhiForCausalLM": ["phi"],
    "FalconForCausalLM": ["falcon"],
    "StableLmForCausalLM": ["stablelm"],
}

# Unsupported architectures for gap analysis
UNSUPPORTED_ARCHITECTURE_TAGS = {
    "BertModel": ["bert"],
    "RobertaModel": ["roberta"],
    "DistilBertModel": ["distilbert"],
    "AlbertModel": ["albert"],
    "XLNetLMHeadModel": ["xlnet"],
    "ElectraModel": ["electra"],
    "DebertaModel": ["deberta"],
    "DebertaV2Model": ["deberta-v2"],
    "MPNetModel": ["mpnet"],
    "T5ForConditionalGeneration": ["t5"],
    "BartForConditionalGeneration": ["bart"],
    "MT5ForConditionalGeneration": ["mt5"],
    "PegasusForConditionalGeneration": ["pegasus"],
    "WhisperForConditionalGeneration": ["whisper"],
    "CLIPModel": ["clip"],
    "ViTModel": ["vit"],
    "Wav2Vec2Model": ["wav2vec2"],
}

SUPPORTED_ARCHITECTURES = list(ARCHITECTURE_TO_TAGS.keys())
UNSUPPORTED_ARCHITECTURES = list(UNSUPPORTED_ARCHITECTURE_TAGS.keys())


def scrape_models_by_tag(
    api,
    architecture: str,
    tags: list[str],
    max_models: int = 500,
    task: str = "text-generation",
) -> list[dict]:
    """Scrape models that match any of the given tags.

    Args:
        api: HuggingFace API client
        architecture: Architecture name for labeling
        tags: List of tags to match (model must have at least one)
        max_models: Maximum number of models to return
        task: Pipeline task to filter by

    Returns:
        List of model info dicts
    """
    logger.info(f"Searching for {architecture} (tags: {tags})...")

    found = []
    checked = 0
    tags_lower = [t.lower() for t in tags]

    try:
        for model in api.list_models(
            task=task,
            sort="downloads",
            direction=-1,
        ):
            checked += 1

            if checked > 50000:  # Safety limit
                break

            # Check if any tag matches
            model_tags_lower = [t.lower() for t in (model.tags or [])]
            if any(tag in model_tags_lower for tag in tags_lower):
                found.append({
                    "model_id": model.id,
                    "downloads": getattr(model, 'downloads', 0) or 0,
                    "likes": getattr(model, 'likes', 0) or 0,
                    "last_modified": model.lastModified.isoformat() if hasattr(model, 'lastModified') and model.lastModified else None,
                    "tags": list(model.tags) if model.tags else [],
                })

                if len(found) >= max_models:
                    break

            if checked % 10000 == 0:
                logger.info(f"  Checked {checked}, found {len(found)}...")

    except Exception as e:
        logger.error(f"Error during search: {e}")

    logger.info(f"  Found {len(found)} {architecture} models")
    return found


def scrape_all_architectures(
    output_dir: Path,
    architectures: dict[str, list[str]] = None,
    unsupported: dict[str, list[str]] = None,
    models_per_arch: int = 500,
) -> tuple[dict, dict]:
    """Scrape HuggingFace for all models of supported architectures.

    Args:
        output_dir: Directory to write JSON data files
        architectures: Dict mapping architecture names to their tags
        unsupported: Dict mapping unsupported architecture names to tags
        models_per_arch: Maximum models to fetch per architecture

    Returns:
        Tuple of (supported_report_dict, gaps_report_dict)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for scraping. "
            "Install it with: pip install huggingface_hub"
        )

    if architectures is None:
        architectures = ARCHITECTURE_TO_TAGS
    if unsupported is None:
        unsupported = UNSUPPORTED_ARCHITECTURE_TAGS

    api = HfApi()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = []
    arch_counts = {}

    # Scrape supported architectures
    logger.info(f"Scraping {len(architectures)} supported architectures...")
    for arch, tags in architectures.items():
        models = scrape_models_by_tag(
            api, arch, tags,
            max_models=models_per_arch,
            task="text-generation",
        )
        arch_counts[arch] = len(models)

        for model in models:
            all_models.append({
                "architecture_id": arch,
                "model_id": model["model_id"],
                "verified": False,
                "verified_date": None,
                "metadata": {
                    "downloads": model["downloads"],
                    "likes": model["likes"],
                    "last_modified": model["last_modified"],
                    "tags": model["tags"],
                },
            })

        time.sleep(0.5)  # Rate limit

    # Build supported models report
    supported_report = {
        "generated_at": date.today().isoformat(),
        "total_architectures": len([a for a in arch_counts if arch_counts[a] > 0]),
        "total_models": len(all_models),
        "total_verified": 0,
        "models": all_models,
    }

    # Write supported models
    supported_path = output_dir / "supported_models.json"
    with open(supported_path, "w") as f:
        json.dump(supported_report, f, indent=2)
    logger.info(f"Wrote {len(all_models)} supported models to {supported_path}")

    # Scrape unsupported architectures for gap analysis
    logger.info(f"Scraping {len(unsupported)} unsupported architectures for gap analysis...")
    gaps = []

    for arch, tags in unsupported.items():
        # For unsupported, just get a count (limit to 100 to estimate)
        # Use appropriate task for each architecture type
        task = "text-generation"
        if any(t in ["bert", "roberta", "distilbert", "albert", "electra", "deberta", "mpnet"] for t in tags):
            task = None  # Search across all tasks for encoder models

        models = scrape_models_by_tag(api, arch, tags, max_models=100, task=task)
        if models:
            gaps.append({
                "architecture_id": arch,
                "total_models": len(models),
            })

        time.sleep(0.3)

    # Sort by count
    gaps.sort(key=lambda x: x["total_models"], reverse=True)

    gaps_report = {
        "generated_at": date.today().isoformat(),
        "total_unsupported": len(gaps),
        "gaps": gaps,
    }

    gaps_path = output_dir / "architecture_gaps.json"
    with open(gaps_path, "w") as f:
        json.dump(gaps_report, f, indent=2)
    logger.info(f"Wrote {len(gaps)} architecture gaps to {gaps_path}")

    # Write empty verification history
    verification_path = output_dir / "verification_history.json"
    with open(verification_path, "w") as f:
        json.dump({
            "last_updated": datetime.now().isoformat(),
            "records": [],
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"  Supported models: {len(all_models)}")
    logger.info(f"  Architectures with models: {len([a for a in arch_counts if arch_counts[a] > 0])}")
    for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"    - {arch}: {count}")
    logger.info(f"  Unsupported architectures tracked: {len(gaps)}")
    logger.info("=" * 60)

    return supported_report, gaps_report


def main():
    parser = argparse.ArgumentParser(
        description="Scrape HuggingFace for TransformerLens-compatible models.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for JSON data files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum models per architecture (default: 500)",
    )

    args = parser.parse_args()

    scrape_all_architectures(
        output_dir=args.output,
        models_per_arch=args.limit,
    )


if __name__ == "__main__":
    main()
