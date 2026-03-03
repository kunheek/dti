"""DTI Metrics Package.

This package provides core metrics for evaluating diffusion models and image generation:
- Image-image similarity using DINOv2 and other feature extractors
- Image-text similarity using SigLIP and CLIP-like models
"""

from .base import BaseMetric
from .dataset import MetricDataset
from .image_sim import (
    CLIPImageSimilarity,
    DINOv2Similarity,
    ImageSimilarityMetric,
)
from .image_text import (
    CLIPScore,
    SigLIPScore,
)

__all__ = [
    "MetricDataset",
    # Base classes
    "BaseMetric",
    # Image-image similarity
    "ImageSimilarityMetric",
    "DINOv2Similarity",
    "CLIPImageSimilarity",
    # Image-text similarity
    "CLIPScore",
    "SigLIPScore",
]
