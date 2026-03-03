from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the base metric.

        Args:
            device: Device to run computations on. If None, will use CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.scores: List[float] = []
        self.reset()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Compute the metric."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset accumulated scores."""
        self.scores = []

    def update(self, score: Union[float, torch.Tensor]) -> None:
        """Update with a new score."""
        if isinstance(score, torch.Tensor):
            if score.numel() == 1:
                score = score.item()
            else:
                score = score.mean().item()
        self.scores.append(float(score))

    def compute(self) -> Dict[str, float]:
        """Compute statistics from accumulated scores."""
        if not self.scores:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        scores_array = np.array(self.scores)
        return {
            "count": len(self.scores),
            "mean": scores_array.mean(),
            "std": scores_array.std(),
            "min": scores_array.min(),
            "max": scores_array.max(),
        }

    def report(self) -> str:
        """Generate a human-readable report of the metrics."""
        results = self.compute()
        lines = []
        for key, value in results.items():
            lines.append(f"{key}: {value:.4f}")
        return "\n".join(lines)

    def summary(self) -> Dict[str, float]:
        """Return a summary of the computed metrics."""
        return self.compute()
