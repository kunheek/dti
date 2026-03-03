from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from PIL import Image


class MetricDataset(torch.utils.data.Dataset):
    """Dataset for evaluation that can apply multiple processors to images."""

    def __init__(
        self,
        sample_dir: str,
        processors: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the evaluation dataset.

        Args:
            sample_dir: Directory containing sample images in SEED/PROMPT.png format
            processors: Dictionary of {name: processor} for preprocessing images
            return_raw: Whether to also return the raw PIL image
            dinov2_processor: DINOv2 image processor (legacy support)
            clip_processor: CLIP image processor (legacy support)
            siglip_processor: SigLIP image processor (legacy support)
        """
        # sample_dir/SEED/PROMPT.png
        self.samples = list(Path(sample_dir).glob("*/*.png"))
        self.processors = processors or {}

        # Validate processors
        for name, processor in self.processors.items():
            if not hasattr(processor, "__call__"):
                raise ValueError(f"Processor '{name}' must be callable")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.

        Returns:
            Dictionary containing:
            - 'prompt': The text prompt
            - 'raw_image': Raw PIL image (if return_raw=True)
            - '{processor_name}': Processed image for each processor
        """
        sample = self.samples[index]
        prompt = sample.name.split(".")[0]
        prompt = f"{prompt.replace('_', ' ')}."
        image = Image.open(str(sample)).convert("RGB")

        result = {"texts": prompt, "images": image}

        for name, processor in self.processors.items():
            inputs = processor(images=image, return_tensors="pt")
            result[name] = inputs.pixel_values
        return result


def collate_eval_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for evaluation batches.

    Args:
        batch: List of dictionaries from dataset

    Returns:
        Batched dictionary with stacked tensors
    """
    if not batch:
        return {}

    # Get all keys from the first item
    keys = batch[0].keys()
    result = {}

    for key in keys:
        if key in ("images", "texts"):
            result[key] = [item[key] for item in batch]
        else:
            # Stack tensors
            result[key] = torch.cat([item[key] for item in batch], dim=0)
    return result
