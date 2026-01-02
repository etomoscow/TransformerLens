"""Model Registry tools for TransformerLens.

This package provides tools for discovering and documenting HuggingFace models
that are compatible with TransformerLens.

Main modules:
    - api: Public API for programmatic access to model registry data
    - schemas: Data classes for model entries, architecture gaps, etc.
    - verification: Verification tracking for model compatibility
    - exceptions: Custom exceptions for the model registry

Example usage:
    >>> from transformer_lens.tools.model_registry import api
    >>> # Check if a model is supported
    >>> api.is_model_supported("gpt2")
    True
    >>> # Get all models for an architecture
    >>> models = api.get_architecture_models("GPT2LMHeadModel")
"""

from .exceptions import (
    ArchitectureNotSupportedError,
    DataNotLoadedError,
    DataValidationError,
    ModelNotFoundError,
    ModelRegistryError,
)
from .schemas import (
    ArchitectureAnalysis,
    ArchitectureGap,
    ArchitectureGapsReport,
    ArchitectureStats,
    ModelEntry,
    ModelMetadata,
    SupportedModelsReport,
)
from .verification import VerificationHistory, VerificationRecord

__all__ = [
    # Exceptions
    "ModelRegistryError",
    "ModelNotFoundError",
    "ArchitectureNotSupportedError",
    "DataNotLoadedError",
    "DataValidationError",
    # Schemas
    "ModelEntry",
    "ModelMetadata",
    "ArchitectureGap",
    "ArchitectureStats",
    "ArchitectureAnalysis",
    "SupportedModelsReport",
    "ArchitectureGapsReport",
    # Verification
    "VerificationRecord",
    "VerificationHistory",
]
