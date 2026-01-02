"""Public API for the TransformerLens model registry.

This module provides a clean, programmatic interface for accessing model registry
data. It supports lazy loading with in-memory caching to avoid repeated file reads.

Example usage:
    >>> from transformer_lens.tools.model_registry import api
    >>>
    >>> # Check if a model is supported
    >>> api.is_model_supported("gpt2")
    True
    >>>
    >>> # Get all supported models
    >>> models = api.get_supported_models()
    >>> len(models)
    1234
    >>>
    >>> # Get models for a specific architecture
    >>> gpt2_models = api.get_architecture_models("GPT2LMHeadModel")
    >>>
    >>> # Get unsupported architectures ranked by model count
    >>> gaps = api.get_unsupported_architectures(min_models=100, top_n=10)
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from .exceptions import (
    ArchitectureNotSupportedError,
    DataNotLoadedError,
    ModelNotFoundError,
)
from .schemas import (
    ArchitectureGap,
    ArchitectureGapsReport,
    ArchitectureStats,
    ModelEntry,
    SupportedModelsReport,
)
from .verification import VerificationHistory

logger = logging.getLogger(__name__)

# Module-level cache for lazy loading
_cache: dict[str, object] = {}
_cache_lock = Lock()

# Default data directory (relative to this module)
_DATA_DIR = Path(__file__).parent / "data"


def _get_data_path(filename: str) -> Path:
    """Get the path to a data file.

    Args:
        filename: Name of the data file

    Returns:
        Path to the data file
    """
    return _DATA_DIR / filename


def _load_json(filename: str) -> dict:
    """Load a JSON file from the data directory.

    Args:
        filename: Name of the JSON file

    Returns:
        Parsed JSON data as a dictionary

    Raises:
        DataNotLoadedError: If the file doesn't exist or can't be read
    """
    path = _get_data_path(filename)
    if not path.exists():
        raise DataNotLoadedError(filename, str(path))
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise DataNotLoadedError(filename, str(path)) from e


def _get_supported_models_report() -> SupportedModelsReport:
    """Get the cached supported models report, loading if necessary.

    Returns:
        The SupportedModelsReport instance

    Raises:
        DataNotLoadedError: If the data file is not available
    """
    cache_key = "supported_models"
    with _cache_lock:
        if cache_key not in _cache:
            data = _load_json("supported_models.json")
            _cache[cache_key] = SupportedModelsReport.from_dict(data)
        return _cache[cache_key]


def _get_architecture_gaps_report() -> ArchitectureGapsReport:
    """Get the cached architecture gaps report, loading if necessary.

    Returns:
        The ArchitectureGapsReport instance

    Raises:
        DataNotLoadedError: If the data file is not available
    """
    cache_key = "architecture_gaps"
    with _cache_lock:
        if cache_key not in _cache:
            data = _load_json("architecture_gaps.json")
            _cache[cache_key] = ArchitectureGapsReport.from_dict(data)
        return _cache[cache_key]


def _get_verification_history() -> VerificationHistory:
    """Get the cached verification history, loading if necessary.

    Returns:
        The VerificationHistory instance

    Raises:
        DataNotLoadedError: If the data file is not available
    """
    cache_key = "verification_history"
    with _cache_lock:
        if cache_key not in _cache:
            data = _load_json("verification_history.json")
            _cache[cache_key] = VerificationHistory.from_dict(data)
        return _cache[cache_key]


def clear_cache() -> None:
    """Clear all cached data.

    This forces data to be reloaded from disk on the next access.
    Useful after updating data files or for testing.
    """
    with _cache_lock:
        _cache.clear()
    logger.debug("Model registry cache cleared")


def get_supported_models(
    architecture: Optional[str] = None,
    verified_only: bool = False,
) -> list[ModelEntry]:
    """Get a list of supported models.

    Args:
        architecture: Filter by architecture ID (e.g., "GPT2LMHeadModel").
            If None, returns all supported models.
        verified_only: If True, only return models that have been verified
            to work with TransformerLens.

    Returns:
        List of ModelEntry objects matching the filters

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> models = get_supported_models(architecture="GPT2LMHeadModel")
        >>> len(models)
        42
        >>> verified = get_supported_models(verified_only=True)
    """
    report = _get_supported_models_report()
    models = report.models

    if architecture:
        models = [m for m in models if m.architecture_id == architecture]

    if verified_only:
        models = [m for m in models if m.verified]

    return models


def get_unsupported_architectures(
    min_models: int = 0,
    top_n: Optional[int] = None,
) -> list[ArchitectureGap]:
    """Get a list of unsupported architectures sorted by model count.

    Args:
        min_models: Minimum number of models for an architecture to be included.
            Useful for filtering out rare architectures.
        top_n: Return only the top N architectures by model count.
            If None, returns all matching architectures.

    Returns:
        List of ArchitectureGap objects sorted by total_models (descending)

    Raises:
        DataNotLoadedError: If the architecture gaps data is not available

    Example:
        >>> # Get top 10 unsupported architectures with 100+ models
        >>> gaps = get_unsupported_architectures(min_models=100, top_n=10)
        >>> for gap in gaps:
        ...     print(f"{gap.architecture_id}: {gap.total_models} models")
    """
    report = _get_architecture_gaps_report()
    gaps = report.gaps

    if min_models > 0:
        gaps = [g for g in gaps if g.total_models >= min_models]

    # Already sorted by total_models descending in the report
    if top_n is not None:
        gaps = gaps[:top_n]

    return gaps


def is_model_supported(model_id: str) -> bool:
    """Check if a model is supported by TransformerLens.

    Args:
        model_id: The HuggingFace model ID to check (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")

    Returns:
        True if the model is in the supported models list, False otherwise

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> is_model_supported("gpt2")
        True
        >>> is_model_supported("some-unsupported-model")
        False
    """
    report = _get_supported_models_report()
    return any(m.model_id == model_id for m in report.models)


def get_model_architecture(model_id: str) -> Optional[str]:
    """Get the architecture ID for a given model.

    Args:
        model_id: The HuggingFace model ID to look up

    Returns:
        The architecture ID (e.g., "GPT2LMHeadModel"), or None if not found

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> get_model_architecture("gpt2")
        'GPT2LMHeadModel'
        >>> get_model_architecture("unknown-model")
        None
    """
    report = _get_supported_models_report()
    for model in report.models:
        if model.model_id == model_id:
            return model.architecture_id
    return None


def get_architecture_models(architecture_id: str) -> list[str]:
    """Get all model IDs for a given architecture.

    Args:
        architecture_id: The architecture to get models for (e.g., "GPT2LMHeadModel")

    Returns:
        List of model IDs that use this architecture

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> models = get_architecture_models("GPT2LMHeadModel")
        >>> "gpt2" in models
        True
    """
    report = _get_supported_models_report()
    return [m.model_id for m in report.models if m.architecture_id == architecture_id]


def suggest_similar_model(model_id: str) -> Optional[str]:
    """Suggest a similar supported model for an unsupported model ID.

    This function attempts to find a supported model that is similar to the
    requested model based on naming patterns. Useful for providing helpful
    suggestions when a user tries to use an unsupported model.

    Args:
        model_id: The model ID that is not supported

    Returns:
        A suggested model ID, or None if no similar model is found

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> suggest_similar_model("bigscience/bloom-560m")
        'bigscience/bloom-1b1'  # A supported BLOOM model
    """
    report = _get_supported_models_report()

    # If the model is already supported, return None (no suggestion needed)
    if any(m.model_id == model_id for m in report.models):
        return None

    # Extract potential matching criteria from the model ID
    model_id_lower = model_id.lower()
    parts = model_id.replace("/", "-").replace("_", "-").lower().split("-")

    # Build a scoring function for similarity
    def score_model(candidate: ModelEntry) -> int:
        candidate_lower = candidate.model_id.lower()
        candidate_parts = candidate.model_id.replace("/", "-").replace("_", "-").lower().split("-")
        score = 0

        # Same organization prefix
        if "/" in model_id and "/" in candidate.model_id:
            if model_id.split("/")[0].lower() == candidate.model_id.split("/")[0].lower():
                score += 10

        # Matching parts
        for part in parts:
            if len(part) > 2 and part in candidate_lower:
                score += 5

        # Architecture name hints
        arch_hints = ["gpt", "llama", "bloom", "opt", "mistral", "gemma", "phi", "qwen"]
        for hint in arch_hints:
            if hint in model_id_lower and hint in candidate_lower:
                score += 8

        return score

    # Score all models and find the best match
    scored = [(m, score_model(m)) for m in report.models]
    scored = [(m, s) for m, s in scored if s > 0]  # Only consider matches with some score
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        return scored[0][0].model_id
    return None


def get_model_info(model_id: str) -> ModelEntry:
    """Get full information about a specific model.

    Args:
        model_id: The HuggingFace model ID to look up

    Returns:
        The ModelEntry for this model

    Raises:
        ModelNotFoundError: If the model is not in the registry
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> info = get_model_info("gpt2")
        >>> info.architecture_id
        'GPT2LMHeadModel'
        >>> info.verified
        True
    """
    report = _get_supported_models_report()
    for model in report.models:
        if model.model_id == model_id:
            return model

    # Model not found - try to suggest an alternative
    suggestion = suggest_similar_model(model_id)
    raise ModelNotFoundError(model_id, suggestion)


def get_supported_architectures() -> list[str]:
    """Get a list of all supported architecture IDs.

    Returns:
        List of unique architecture IDs that TransformerLens supports

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> archs = get_supported_architectures()
        >>> "GPT2LMHeadModel" in archs
        True
    """
    report = _get_supported_models_report()
    return list(sorted(set(m.architecture_id for m in report.models)))


def get_all_architectures_with_stats() -> list[ArchitectureStats]:
    """Get statistics for all architectures (both supported and unsupported).

    Returns:
        List of ArchitectureStats objects for all known architectures,
        sorted by model count (descending)

    Raises:
        DataNotLoadedError: If the registry data is not available

    Example:
        >>> stats = get_all_architectures_with_stats()
        >>> for s in stats[:5]:
        ...     status = "supported" if s.is_supported else "unsupported"
        ...     print(f"{s.architecture_id}: {s.model_count} models ({status})")
    """
    supported_report = _get_supported_models_report()
    gaps_report = _get_architecture_gaps_report()

    # Build stats for supported architectures
    arch_stats: dict[str, ArchitectureStats] = {}

    for model in supported_report.models:
        arch_id = model.architecture_id
        if arch_id not in arch_stats:
            arch_stats[arch_id] = ArchitectureStats(
                architecture_id=arch_id,
                is_supported=True,
                model_count=0,
                verified_count=0,
                example_models=[],
            )
        stats = arch_stats[arch_id]
        stats.model_count += 1
        if model.verified:
            stats.verified_count += 1
        if len(stats.example_models) < 5:
            stats.example_models.append(model.model_id)

    # Add stats for unsupported architectures
    for gap in gaps_report.gaps:
        if gap.architecture_id not in arch_stats:
            arch_stats[gap.architecture_id] = ArchitectureStats(
                architecture_id=gap.architecture_id,
                is_supported=False,
                model_count=gap.total_models,
                verified_count=0,
                example_models=[],
            )

    # Sort by model count descending
    result = sorted(arch_stats.values(), key=lambda x: x.model_count, reverse=True)
    return result


def is_architecture_supported(architecture_id: str) -> bool:
    """Check if an architecture is supported by TransformerLens.

    Args:
        architecture_id: The architecture ID to check

    Returns:
        True if the architecture is supported, False otherwise

    Raises:
        DataNotLoadedError: If the supported models data is not available

    Example:
        >>> is_architecture_supported("GPT2LMHeadModel")
        True
        >>> is_architecture_supported("SomeUnknownModel")
        False
    """
    report = _get_supported_models_report()
    return any(m.architecture_id == architecture_id for m in report.models)


def get_registry_stats() -> dict:
    """Get summary statistics about the model registry.

    Returns:
        Dictionary with registry statistics including:
        - total_supported_models: Number of supported models
        - total_supported_architectures: Number of supported architectures
        - total_verified: Number of verified models
        - total_unsupported_architectures: Number of unsupported architectures
        - generated_at: When the data was generated

    Raises:
        DataNotLoadedError: If the registry data is not available

    Example:
        >>> stats = get_registry_stats()
        >>> print(f"Supported: {stats['total_supported_models']} models")
    """
    supported = _get_supported_models_report()
    gaps = _get_architecture_gaps_report()

    return {
        "total_supported_models": supported.total_models,
        "total_supported_architectures": supported.total_architectures,
        "total_verified": supported.total_verified,
        "total_unsupported_architectures": gaps.total_unsupported,
        "supported_generated_at": supported.generated_at.isoformat(),
        "gaps_generated_at": gaps.generated_at.isoformat(),
    }
