"""Image-text similarity metrics using vision-language models."""

from typing import Optional, Union, List
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image

from .base import BaseMetric


class CLIPScore(BaseMetric):
    """
    CLIPScore: CLIP-based image-text similarity metric using cosine similarity.

    Based on the paper "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    Uses cosine similarity between CLIP image and text embeddings.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize CLIPScore metric.

        Args:
            model_name: CLIP model name.
            device: Device to run computations on.
        """
        super().__init__(device)
        self.model_name = model_name

        # Load CLIP processor and model
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        # Freeze the model parameters
        self.model.eval().requires_grad_(False)

    def __call__(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        texts: List[str],
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Compute CLIP similarity between images and text prompts.

        Args:
            images: List of PIL images or tensor (N images).
            texts: List of text prompts (M texts).
            aggregate: How to aggregate the N×M similarity matrix:
                     - "mean": Average over all pairs
                     - "max_per_image": Maximum similarity for each image (N,)
                     - "max_per_text": Maximum similarity for each text (M,)
                     - "none": Return full N×M matrix
                     - "row_mean": Average similarity for each image (N,)
                     - "col_mean": Average similarity for each text (M,)

        Returns:
            Similarity scores based on aggregation method.
        """
        # Handle different input types
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            # Convert tensor to PIL images for processor
            images = [
                Image.fromarray(
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                for img in images
            ]

        # Process inputs
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # CLIP returns logits which are already similarity scores
            if hasattr(outputs, "logits_per_image"):
                # Apply cosine similarity (CLIP's default)
                similarity_matrix = outputs.logits_per_image  # (N, M)
            else:
                # Fallback: extract features and compute cosine similarity
                image_features = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
                text_features = self.model.get_text_features(
                    input_ids=inputs["input_ids"]
                )

                # Normalize and compute cosine similarity
                image_features = F.normalize(image_features, p=2, dim=-1)  # (N, D)
                text_features = F.normalize(text_features, p=2, dim=-1)  # (M, D)
                similarity_matrix = torch.mm(
                    image_features, text_features.t()
                )  # (N, M)

        # Aggregate based on the specified method
        if aggregate == "mean":
            score = similarity_matrix.mean()
        elif aggregate == "max_per_image":
            score = similarity_matrix.max(dim=1)[0]  # Max over texts for each image
        elif aggregate == "max_per_text":
            score = similarity_matrix.max(dim=0)[0]  # Max over images for each text
        elif aggregate == "none":
            score = similarity_matrix
        elif aggregate == "row_mean":
            score = similarity_matrix.mean(dim=1)  # Mean over texts for each image
        elif aggregate == "col_mean":
            score = similarity_matrix.mean(dim=0)  # Mean over images for each text
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregate}. "
                f"Supported: 'mean', 'max_per_image', 'max_per_text', 'none', 'row_mean', 'col_mean'"
            )

        # Update metrics (for backward compatibility, use mean for internal tracking)
        self.update(similarity_matrix.mean())

        return score


class SigLIPScore(BaseMetric):
    """
    SigLIPScore: SigLIP-based image-text similarity metric using sigmoid on logits.

    Uses the SigLIP model's training objective which applies sigmoid to the
    dot product similarity (logits) instead of cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize SigLIPScore metric.

        Args:
            model_name: SigLIP model name.
            device: Device to run computations on.
        """
        super().__init__(device)
        self.model_name = model_name

        # Load SigLIP processor and model.
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Freeze the model parameters
        self.model.eval().requires_grad_(False)

    def __call__(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        texts: List[str],
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Compute SigLIP similarity between images and text prompts.

        Args:
            images: List of PIL images or tensor (N images).
            texts: List of text prompts (M texts).
            aggregate: How to aggregate the N×M similarity matrix:
                     - "mean": Average over all pairs
                     - "max_per_image": Maximum similarity for each image (N,)
                     - "max_per_text": Maximum similarity for each text (M,)
                     - "none": Return full N×M matrix
                     - "row_mean": Average similarity for each image (N,)
                     - "col_mean": Average similarity for each text (M,)

        Returns:
            Similarity scores based on aggregation method.
        """
        # Handle different input types
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            # Convert tensor to PIL images for processor
            images = [
                Image.fromarray(
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                for img in images
            ]

        # For SigLIP, follow the pipeline template to get same results as in the example
        # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
        inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            # SigLIP returns logits_per_image which should be used with sigmoid
            if hasattr(outputs, "logits_per_image"):
                logits_per_image = outputs.logits_per_image  # (N, M)
                # Apply sigmoid as per SigLIP's training objective
                similarity_matrix = torch.sigmoid(logits_per_image)
            else:
                # Fallback: extract features and compute sigmoid similarity
                image_features = (
                    outputs.image_embeds
                    if hasattr(outputs, "image_embeds")
                    else outputs[0]
                )
                text_features = (
                    outputs.text_embeds
                    if hasattr(outputs, "text_embeds")
                    else outputs[1]
                )

                # Compute dot products and apply sigmoid
                similarity_matrix = torch.mm(
                    image_features, text_features.t()
                )  # (N, M)
                similarity_matrix = torch.sigmoid(similarity_matrix)

        # Aggregate based on the specified method
        if aggregate == "diag":
            score = torch.diagonal(
                similarity_matrix
            )  # Extract diagonal for paired inputs.
        elif aggregate == "mean":
            score = similarity_matrix.mean()
        elif aggregate == "max_per_image":
            score = similarity_matrix.max(dim=1)[0]  # Max over texts for each image
        elif aggregate == "max_per_text":
            score = similarity_matrix.max(dim=0)[0]  # Max over images for each text
        elif aggregate == "none":
            score = similarity_matrix
        elif aggregate == "row_mean":
            score = similarity_matrix.mean(dim=1)  # Mean over texts for each image
        elif aggregate == "col_mean":
            score = similarity_matrix.mean(dim=0)  # Mean over images for each text
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregate}. "
                f"Supported: 'mean', 'max_per_image', 'max_per_text', 'none', 'row_mean', 'col_mean'"
            )

        # Update metrics (for backward compatibility, use mean for internal tracking)
        self.update(similarity_matrix.mean())

        return score
