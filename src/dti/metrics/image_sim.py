"""Image-image similarity metrics using various feature extractors."""

from typing import Optional, Union, List
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from .base import BaseMetric


class ImageSimilarityMetric(BaseMetric):
    """Compute similarity between generated images and reference images using feature extractors."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[Union[str, torch.device]] = None,
        similarity_metric: str = "cosine",
    ):
        """
        Initialize the image similarity metric.

        Args:
            model_name: Name of the feature extractor model. Default is DINOv2.
                       Other options: "facebook/dinov2-small", "facebook/dinov2-large",
                       "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"
            device: Device to run computations on.
            similarity_metric: Similarity metric to use ("cosine", "l2", "l1").
        """
        super().__init__(device)
        self.model_name = model_name
        self.similarity_metric = similarity_metric

        # Load the feature extractor.
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Freeze the model parameters.
        self.model.eval().requires_grad_(False)

    def _extract_features(
        self, images: Union[List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        """Extract features from images."""
        if isinstance(images, torch.Tensor):
            # Assume images are already preprocessed tensors.
            if images.dim() == 3:
                images = images.unsqueeze(0)
            inputs = {"pixel_values": images.to(self.device)}
        else:
            # Process PIL images
            inputs = self.processor(images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Handle different model outputs
            if hasattr(outputs, "last_hidden_state"):
                # For DINOv2 and similar models
                features = outputs.last_hidden_state
                # Use CLS token or global average pooling
                if features.size(1) > 1:  # Has CLS token
                    features = features[:, 0]  # Use CLS token
                else:
                    features = features.mean(dim=1)  # Global average pooling
            elif hasattr(outputs, "pooler_output"):
                # For CLIP-like models
                features = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                # For some CLIP variants
                features = outputs.image_embeds
            else:
                # Fallback: use the first element if it's a tensor
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    features = outputs[0]
                    if features.dim() > 2:
                        features = features.mean(
                            dim=tuple(range(1, features.dim() - 1))
                        )
                else:
                    raise ValueError(
                        f"Unknown output format from model {self.model_name}"
                    )

        return features

    def _compute_similarity(
        self, features1: torch.Tensor, features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two feature tensors.

        Args:
            features1: Generated image features of shape (N, D)
            features2: Reference image features of shape (M, D)

        Returns:
            Similarity matrix of shape (N, M) where entry (i, j) is the similarity
            between generated image i and reference image j.
        """
        features1 = features1.to(dtype=torch.float32)
        features2 = features2.to(dtype=torch.float32)
        if self.similarity_metric == "cosine":
            # Normalize features
            features1 = F.normalize(features1, p=2, dim=-1)  # (N, D)
            features2 = F.normalize(features2, p=2, dim=-1)  # (M, D)
            # Compute cosine similarity matrix: (N, D) @ (D, M) = (N, M)
            similarity = torch.mm(features1, features2.t())
        elif self.similarity_metric == "l2":
            # Compute negative L2 distance matrix (higher is more similar)
            # Expand dimensions for broadcasting: (N, 1, D) and (1, M, D)
            features1_expanded = features1.unsqueeze(1)  # (N, 1, D)
            features2_expanded = features2.unsqueeze(0)  # (1, M, D)
            # Compute L2 distance for all pairs
            similarity = -torch.norm(
                features1_expanded - features2_expanded, p=2, dim=-1
            )  # (N, M)
        elif self.similarity_metric == "l1":
            # Compute negative L1 distance matrix (higher is more similar)
            # Expand dimensions for broadcasting: (N, 1, D) and (1, M, D)
            features1_expanded = features1.unsqueeze(1)  # (N, 1, D)
            features2_expanded = features2.unsqueeze(0)  # (1, M, D)
            # Compute L1 distance for all pairs
            similarity = -torch.norm(
                features1_expanded - features2_expanded, p=1, dim=-1
            )  # (N, M)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return similarity

    def __call__(
        self,
        generated_images: Union[List[Image.Image], torch.Tensor],
        reference_images: Union[List[Image.Image], torch.Tensor],
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Compute similarity between generated and reference images.

        Args:
            generated_images: List of generated PIL images or tensor (N images).
            reference_images: List of reference PIL images or tensor (M images).
            aggregate: How to aggregate the N×M similarity matrix:
                     - "mean": Average over all pairs
                     - "max": Maximum similarity for each generated image (N,)
                     - "none": Return full N×M matrix
                     - "row_mean": Average similarity for each generated image (N,)
                     - "col_mean": Average similarity for each reference image (M,)

        Returns:
            Similarity scores as a tensor:
            - If aggregate="mean": Single scalar value
            - If aggregate="max": Tensor of shape (N,) with max similarity for each generated image
            - If aggregate="none": Full similarity matrix of shape (N, M)
            - If aggregate="row_mean": Tensor of shape (N,) with mean similarity for each generated image
            - If aggregate="col_mean": Tensor of shape (M,) with mean similarity for each reference image
        """
        # Extract features.
        gen_features = self._extract_features(generated_images)  # (N, D)
        ref_features = self._extract_features(reference_images)  # (M, D)

        # Compute similarity matrix (N, M)
        similarity_matrix = self._compute_similarity(gen_features, ref_features)

        # Aggregate based on the specified method
        if aggregate == "mean":
            score = similarity_matrix.mean()
        elif aggregate == "max":
            score = similarity_matrix.max(dim=1)[
                0
            ]  # Max over reference images for each generated image
        elif aggregate == "none":
            score = similarity_matrix
        elif aggregate == "row_mean":
            score = similarity_matrix.mean(
                dim=1
            )  # Mean over reference images for each generated image
        elif aggregate == "col_mean":
            score = similarity_matrix.mean(
                dim=0
            )  # Mean over generated images for each reference image
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregate}. "
                f"Supported: 'mean', 'max', 'none', 'row_mean', 'col_mean'"
            )

        # Update metrics (for backward compatibility, use mean for internal tracking)
        self.update(similarity_matrix.mean())

        return score


class DINOv2Similarity(ImageSimilarityMetric):
    """DINOv2-based image similarity metric."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[Union[str, torch.device]] = None,
        similarity_metric: str = "cosine",
    ):
        """
        Initialize DINOv2 similarity metric.

        Args:
            model_size: Size of DINOv2 model ("small", "base", "large", "giant").
            device: Device to run computations on.
            similarity_metric: Similarity metric to use.
        """
        model_name = f"facebook/dinov2-{model_size}"
        super().__init__(model_name, device, similarity_metric)


class CLIPImageSimilarity(ImageSimilarityMetric):
    """CLIP-based image similarity metric."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
        similarity_metric: str = "cosine",
    ):
        """
        Initialize CLIP image similarity metric.

        Args:
            model_name: CLIP model name.
            device: Device to run computations on.
            similarity_metric: Similarity metric to use.
        """
        super().__init__(model_name, device, similarity_metric)
