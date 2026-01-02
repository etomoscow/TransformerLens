"""Data schemas for the model registry.

This module defines the dataclasses used throughout the model registry for
representing supported models, architecture gaps, and related metadata.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class ModelMetadata:
    """Metadata for a model from HuggingFace.

    Attributes:
        downloads: Total download count for the model
        likes: Number of likes/stars on HuggingFace
        last_modified: When the model was last updated
        tags: List of tags associated with the model
        parameter_count: Estimated number of parameters (if available)
    """

    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    parameter_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "downloads": self.downloads,
            "likes": self.likes,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "tags": self.tags,
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from a dictionary."""
        last_modified = None
        if data.get("last_modified"):
            last_modified = datetime.fromisoformat(data["last_modified"])
        return cls(
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            last_modified=last_modified,
            tags=data.get("tags", []),
            parameter_count=data.get("parameter_count"),
        )


@dataclass
class ModelEntry:
    """A single model entry in the supported models list.

    Attributes:
        architecture_id: The architecture type (e.g., "GPT2LMHeadModel")
        model_id: The HuggingFace model ID (e.g., "gpt2", "openai-community/gpt2")
        verified: Whether this model has been verified to work with TransformerLens
        verified_date: Date when verification was performed
        metadata: Optional metadata from HuggingFace
    """

    architecture_id: str
    model_id: str
    verified: bool = False
    verified_date: Optional[date] = None
    metadata: Optional[ModelMetadata] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "model_id": self.model_id,
            "verified": self.verified,
            "verified_date": self.verified_date.isoformat() if self.verified_date else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        """Create from a dictionary."""
        verified_date = None
        if data.get("verified_date"):
            verified_date = date.fromisoformat(data["verified_date"])
        metadata = None
        if data.get("metadata"):
            metadata = ModelMetadata.from_dict(data["metadata"])
        return cls(
            architecture_id=data["architecture_id"],
            model_id=data["model_id"],
            verified=data.get("verified", False),
            verified_date=verified_date,
            metadata=metadata,
        )


@dataclass
class ArchitectureGap:
    """An unsupported architecture with model count.

    Attributes:
        architecture_id: The architecture type not supported by TransformerLens
        total_models: Number of models on HuggingFace using this architecture
    """

    architecture_id: str
    total_models: int

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "total_models": self.total_models,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureGap":
        """Create from a dictionary."""
        return cls(
            architecture_id=data["architecture_id"],
            total_models=data["total_models"],
        )


@dataclass
class SupportedModelsReport:
    """Report containing all supported models.

    Attributes:
        generated_at: Date when this report was generated
        total_architectures: Number of unique supported architectures
        total_models: Total number of supported models
        total_verified: Number of models that have been verified
        models: List of all model entries
    """

    generated_at: date
    total_architectures: int
    total_models: int
    total_verified: int
    models: list[ModelEntry]

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_architectures": self.total_architectures,
            "total_models": self.total_models,
            "total_verified": self.total_verified,
            "models": [m.to_dict() for m in self.models],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SupportedModelsReport":
        """Create from a dictionary."""
        return cls(
            generated_at=date.fromisoformat(data["generated_at"]),
            total_architectures=data["total_architectures"],
            total_models=data["total_models"],
            total_verified=data["total_verified"],
            models=[ModelEntry.from_dict(m) for m in data["models"]],
        )


@dataclass
class ArchitectureGapsReport:
    """Report containing unsupported architectures.

    Attributes:
        generated_at: Date when this report was generated
        total_unsupported: Number of unsupported architectures
        gaps: List of architecture gaps sorted by model count
    """

    generated_at: date
    total_unsupported: int
    gaps: list[ArchitectureGap]

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_unsupported": self.total_unsupported,
            "gaps": [g.to_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureGapsReport":
        """Create from a dictionary."""
        return cls(
            generated_at=date.fromisoformat(data["generated_at"]),
            total_unsupported=data["total_unsupported"],
            gaps=[ArchitectureGap.from_dict(g) for g in data["gaps"]],
        )


@dataclass
class ArchitectureStats:
    """Statistics about an architecture including supported and gap info.

    Attributes:
        architecture_id: The architecture identifier
        is_supported: Whether TransformerLens supports this architecture
        model_count: Number of models using this architecture
        verified_count: Number of verified models (if supported)
        example_models: Sample model IDs for this architecture
    """

    architecture_id: str
    is_supported: bool
    model_count: int
    verified_count: int = 0
    example_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "is_supported": self.is_supported,
            "model_count": self.model_count,
            "verified_count": self.verified_count,
            "example_models": self.example_models,
        }


@dataclass
class ArchitectureAnalysis:
    """Analysis result for prioritizing architecture support.

    Attributes:
        architecture_id: The architecture identifier
        total_models: Total models using this architecture
        total_downloads: Sum of downloads across all models
        priority_score: Computed priority score for implementation
        top_models: Most popular models for this architecture
    """

    architecture_id: str
    total_models: int
    total_downloads: int
    priority_score: float
    top_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "total_models": self.total_models,
            "total_downloads": self.total_downloads,
            "priority_score": self.priority_score,
            "top_models": self.top_models,
        }
