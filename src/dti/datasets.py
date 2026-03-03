"""
Base dataset classes for DTI training.

This module provides base dataset classes that can be inherited by specific training scripts.
The classes handle common functionality like image loading, prompt creation, and transformations.
"""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .constants import (
    PIL_INTERPOLATION,
    imagenet_templates_small,
    imagenet_style_templates_small,
)
from .utils import generate_exponential_samples_in_range


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Base dataset class with common functionality for DTI training.

    This class provides:
    - Image loading from given paths
    - Prompt creation based on templates
    - Basic image transformations (resize, crop, flip)
    - Support for both directory-based and JSON-based data sources
    """

    def __init__(
        self,
        data_root: str,
        instance: Optional[str] = None,
        learnable_property: str = "object",  # [object, style]
        size: int = 512,
        repeats: int = 100,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
        placeholder_token: str = "*",
        center_crop: bool = False,
        zero_pad: bool = False,
    ):
        """
        Initialize the base dataset.

        Args:
            data_root: Path to data directory or JSON file
            instance: Instance name when using JSON data source
            learnable_property: Type of learning ("object" or "style")
            size: Target image size
            repeats: Number of repeats for training set
            interpolation: Interpolation method for resizing
            flip_p: Probability of horizontal flip
            placeholder_token: Token to be replaced in templates
            center_crop: Whether to use center crop instead of random crop
            zero_pad: Whether to add zero padding tokens
        """
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.zero_pad = zero_pad
        self.paired = False  # Whether dataset has paired templates

        # Load data paths and templates
        self._load_data_paths_and_templates(data_root, instance)

        # Set dataset length
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats

        # Setup image transformations
        self._setup_transforms(interpolation)

    def _load_data_paths_and_templates(self, data_root: str, instance: Optional[str]):
        """Load image paths and templates from data root."""
        if data_root.endswith(".json"):
            assert instance is not None, (
                "If data_root is a json file, instance must be specified"
            )
            with open(data_root, "r") as f:
                data = json.load(f)[instance]

            # Load image paths
            image_paths = []
            templates = []
            if "qwen" in data_root:
                for sample in data["images"]:
                    image_paths.append(Path(sample["image_path"]))
                    templates.append(sample["description"])
            else:
                image_paths.append(data["path"])
                templates.append(data["template"])
            self.image_paths = image_paths
            self.templates = templates
            self.paired = True
        else:
            self.data_root = data_root
            self.image_paths = list(Path(self.data_root).iterdir())
            self.templates = self._get_default_templates()

    def _get_default_templates(self) -> List[str]:
        """Get default templates based on learnable property."""
        return (
            imagenet_style_templates_small
            if self.learnable_property == "style"
            else imagenet_templates_small
        )

    def _setup_transforms(self, interpolation: str):
        """Setup image transformations."""
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.resize = transforms.Resize(self.size, interpolation=self.interpolation)
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.crop = (
            transforms.CenterCrop(self.size)
            if self.center_crop
            else transforms.RandomCrop(self.size)
        )

    def __len__(self) -> int:
        return self._length

    def load_image(self, image_path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def create_prompt(self, image_id: int) -> str:
        """Create prompt from template."""
        placeholder_string = self.placeholder_token
        if self.paired:
            text = self.templates[image_id].format(placeholder_string)
        else:
            text = random.choice(self.templates).format(placeholder_string)

        # Add zero padding if enabled
        if self.zero_pad:
            num_zeros = generate_exponential_samples_in_range(
                scale=2,
                size=1,
                low=0,
                high=15,
            )[0]
            if num_zeros > 0:
                pad = " ".join(["<pad>"] * num_zeros)
                text = f"{pad} {text}"

        return text

    def apply_image_transforms(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Apply image transformations and return processed image with crop coordinates.

        Returns:
            Tuple of (processed_image_tensor, crop_top_left_coordinates)
        """
        # Resize image
        image = self.resize(image)

        # Apply cropping and get coordinates
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
            image = transforms.functional.crop(image, y1, x1, h, w)

        # Apply horizontal flip
        image = self.flip_transform(image)

        # Convert to tensor and normalize
        img_array = np.array(image).astype(np.uint8)
        img_array = (img_array / 127.5 - 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        return image_tensor, (y1, x1)

    @abstractmethod
    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Get dataset item. Must be implemented by subclasses.

        This method should handle tokenization and any model-specific processing.
        """
        pass


class TextualInversionDataset(BaseDataset):
    """
    Standard textual inversion dataset for single tokenizer models (SD 1.5, SD 2.1).
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        instance: Optional[str] = None,
        learnable_property: str = "object",
        size: int = 512,
        repeats: int = 100,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
        placeholder_token: str = "*",
        center_crop: bool = False,
        zero_pad: bool = False,
    ):
        """Initialize textual inversion dataset with single tokenizer."""
        super().__init__(
            data_root=data_root,
            instance=instance,
            learnable_property=learnable_property,
            size=size,
            repeats=repeats,
            interpolation=interpolation,
            flip_p=flip_p,
            placeholder_token=placeholder_token,
            center_crop=center_crop,
            zero_pad=zero_pad,
        )
        self.tokenizer = tokenizer

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Get dataset item with single tokenizer processing."""
        example = {}
        image_id = i % self.num_images

        # Load image
        image = self.load_image(self.image_paths[image_id])
        example["original_size"] = (image.height, image.width)

        # Create prompt
        text = self.create_prompt(image_id)

        # Apply image transformations
        image_tensor, crop_coords = self.apply_image_transforms(image)
        example["pixel_values"] = image_tensor
        example["crop_top_left"] = crop_coords

        # Tokenize text
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Add timesteps for diffusion training
        example["timesteps"] = torch.randint(0, 1000, size=(1,), dtype=torch.long)

        return example


class SdxlDataset(BaseDataset):
    """
    Textual inversion dataset for dual tokenizer models (SDXL, SD3).
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_1,
        tokenizer_2,
        instance: Optional[str] = None,
        learnable_property: str = "object",
        size: int = 1024,
        repeats: int = 100,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
        placeholder_token: str = "*",
        center_crop: bool = False,
        zero_pad: bool = False,
    ):
        """Initialize dataset with dual tokenizers."""
        super().__init__(
            data_root=data_root,
            instance=instance,
            learnable_property=learnable_property,
            size=size,
            repeats=repeats,
            interpolation=interpolation,
            flip_p=flip_p,
            placeholder_token=placeholder_token,
            center_crop=center_crop,
            zero_pad=zero_pad,
        )
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Get dataset item with dual tokenizer processing."""
        example = {}
        image_id = i % self.num_images

        # Load image
        image = self.load_image(self.image_paths[image_id])
        example["original_size"] = (image.height, image.width)

        # Create prompt
        text = self.create_prompt(image_id)

        # Apply image transformations
        image_tensor, crop_coords = self.apply_image_transforms(image)
        example["pixel_values"] = image_tensor
        example["crop_top_left"] = crop_coords

        # Tokenize with both tokenizers
        example["input_ids_1"] = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Add timesteps for diffusion training
        example["timesteps"] = torch.randint(0, 1000, size=(1,), dtype=torch.long)

        return example


class SanaDataset(BaseDataset):
    """
    Dataset for SANA model training.
    """

    def __init__(
        self,
        data_root: str,
        instance: Optional[str] = None,
        learnable_property: str = "object",
        size: int = 1024,
        repeats: int = 100,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
        placeholder_token: str = "*",
        center_crop: bool = False,
        zero_pad: bool = False,
    ):
        """Initialize SANA dataset."""
        super().__init__(
            data_root=data_root,
            instance=instance,
            learnable_property=learnable_property,
            size=size,
            repeats=repeats,
            interpolation=interpolation,
            flip_p=flip_p,
            placeholder_token=placeholder_token,
            center_crop=center_crop,
            zero_pad=zero_pad,
        )

    def create_prompt(self, image_id: int) -> str:
        """Create prompt with SANA-specific zero padding format."""
        placeholder_string = self.placeholder_token
        if self.paired:
            text = self.templates[image_id].format(placeholder_string)
        else:
            text = random.choice(self.templates).format(placeholder_string)

        # SANA uses a different zero padding format
        if self.zero_pad:
            num_zeros = generate_exponential_samples_in_range(
                scale=2,
                size=1,
                low=0,
                high=15,
            )[0]
            if num_zeros > 0:
                pad = "".join(["<zero-pad>"] * num_zeros)
                text = f"{pad}{text}"

        return text

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Get dataset item for SANA model."""
        sample = {}
        image_id = i % self.num_images

        # Load image
        image = self.load_image(self.image_paths[image_id])

        # Create prompt
        text = self.create_prompt(image_id)
        sample["prompt"] = text

        # Apply image transformations
        image_tensor, _ = self.apply_image_transforms(image)
        sample["pixel_values"] = image_tensor

        return sample


class LoRADataset(BaseDataset):
    """
    Dataset for LoRA training with additional image token support.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_1,
        tokenizer_2,
        learnable_property: str = "object",
        size: int = 512,
        repeats: int = 100,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
        placeholder_token: str = "*",
        center_crop: bool = False,
        zero_pad: bool = False,
    ):
        """Initialize LoRA dataset with image token support."""
        super().__init__(
            data_root=data_root,
            instance=None,  # LoRA dataset doesn't use JSON format
            learnable_property=learnable_property,
            size=size,
            repeats=repeats,
            interpolation=interpolation,
            flip_p=flip_p,
            placeholder_token=placeholder_token,
            center_crop=center_crop,
            zero_pad=zero_pad,
        )
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

    def create_prompt(self, image_id: int) -> str:
        """Create prompt with optional image tokens."""
        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        # Add zero padding
        if self.zero_pad:
            num_zeros = generate_exponential_samples_in_range(
                scale=2,
                size=1,
                low=0,
                high=15,
            )[0]
            if num_zeros > 0:
                pad = " ".join(["<pad>"] * num_zeros)
                text = f"{pad} {text}"

        return text

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Get dataset item for LoRA training."""
        example = {}
        image_id = i % self.num_images

        # Load image
        image = self.load_image(self.image_paths[image_id])
        example["original_size"] = (image.height, image.width)

        # Create prompt
        text = self.create_prompt(image_id)

        # Apply image transformations
        image_tensor, crop_coords = self.apply_image_transforms(image)
        example["pixel_values"] = image_tensor
        example["crop_top_left"] = crop_coords

        # Tokenize with both tokenizers
        example["input_ids_1"] = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example
