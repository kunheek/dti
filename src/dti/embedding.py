from typing import Optional

import torch
import torch.nn as nn

# Constants
SCALE_TYPES = {"min", "mean", "max"}


class DtiTokenEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        use_scale: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            *args,
            **kwargs,
        )
        self.original_vocab_size = num_embeddings
        self.use_scale = use_scale
        self.scale_cache = {}

        # Separate embedding for added/novel tokens
        self.added_tokens_embeddings = nn.Embedding(0, embedding_dim)
        if self.use_scale:
            self.scales = nn.Embedding(0, 1)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        # Separate handling for original and added tokens
        index_added = input_ids >= self.original_vocab_size

        # Get embeddings for original tokens
        inputs_embeds = super().forward(input_ids)

        # Replace embeddings for added tokens
        if index_added.any():
            added_ids = self.to_added_token_id(input_ids[index_added])
            added_embeds = self.added_tokens_embeddings(added_ids)

            # Apply scaling if enabled
            if self.use_scale:
                added_scales = self.scales(added_ids)
                added_embeds = added_scales * added_embeds

            inputs_embeds[index_added] = added_embeds

        return inputs_embeds

    @torch.no_grad()
    def _compute_embedding_scales(self) -> torch.Tensor:
        """Compute norms of all embeddings, excluding zeros."""
        scales = torch.norm(self.weight, dim=-1, keepdim=True)
        return scales[scales > 0]

    @torch.no_grad()
    def get_scale_stat(self, stat: str = "mean") -> float:
        """Get a statistic of the embedding scales.

        Args:
            stat: Statistic to compute ("mean", "min", "max").

        Returns:
            The computed statistic value.
        """
        if stat not in SCALE_TYPES:
            raise ValueError(f"Unsupported stat: {stat}. Must be one of {SCALE_TYPES}")
        if stat not in self.scale_cache:
            scales = self._compute_embedding_scales()
            self.scale_cache["min"] = scales.min().item()
            self.scale_cache["mean"] = scales.mean().item()
            self.scale_cache["max"] = scales.max().item()
        return self.scale_cache[stat]

    def to_added_token_id(
        self, token_id: int | list[int] | torch.LongTensor
    ) -> int | list[int] | torch.LongTensor:
        """Convert global token ID(s) to added token ID(s).

        Args:
            token_id: Global token ID (int), list of IDs, or tensor of IDs (>= original_vocab_size).

        Returns:
            Added token ID(s) in the same type as input (token_id - original_vocab_size).
        """
        if isinstance(token_id, torch.Tensor):
            return token_id - self.original_vocab_size
        elif isinstance(token_id, list):
            return [tid - self.original_vocab_size for tid in token_id]
        elif isinstance(token_id, int):
            return token_id - self.original_vocab_size
        else:
            raise TypeError(
                f"Unsupported type for token_id: {type(token_id)}. Must be int, list, or torch.Tensor."
            )

    def write_token_embedding(self, token_id: int, embedding: torch.Tensor) -> None:
        """Write embedding to the appropriate storage (added_tokens_embeddings).

        Args:
            token_id: Global token ID.
            embedding: Embedding tensor to write.
        """
        added_id = self.to_added_token_id(token_id)
        self.added_tokens_embeddings.weight.data[added_id] = embedding

    def get_learned_embeddings(
        self, token_ids: list[int], clamp_min: float = 1e-6
    ) -> torch.Tensor:
        """Get embeddings for specified tokens with scales applied.

        Args:
            List of global token IDs.

        Returns:
            Tensor of learned embeddings with scales applied.
        """
        added_ids = self.to_added_token_id(token_ids)
        embeds = self.added_tokens_embeddings.weight[added_ids]

        if self.use_scale:
            scales = self.scales.weight[added_ids]
            norms = torch.norm(embeds, dim=-1, keepdim=True).clamp(min=clamp_min)
            embeds = scales * (embeds / norms)

        return embeds.detach().float()

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        """Resize token embeddings, added tokens, and scales.

        Unified API for V2: resizes original weight (with zeros), added_tokens_embeddings, and scales.

        Args:
            new_num_tokens: New vocabulary size (total tokens).
            pad_to_multiple_of: Optional padding to multiple.
        """
        if pad_to_multiple_of is not None:
            new_num_tokens = (
                (new_num_tokens + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        # Resize all components
        self.resize_original_embeddings(new_num_tokens)
        self.resize_added_token_embeddings(new_num_tokens)

    def resize_token_scales(self, new_num_tokens: int) -> None:
        if not self.use_scale:
            return

        new_num_tokens = new_num_tokens - self.original_vocab_size

        # Update token scale embeddings.
        # This is needed because the token scale embeddings are not tied to the token.
        scales = self.scales.weight.data
        old_num_tokens, old_dim = scales.shape
        if new_num_tokens is None:
            new_num_tokens = old_num_tokens
        if new_num_tokens > old_num_tokens:
            scales = torch.cat(
                [scales, scales.new_zeros(new_num_tokens - old_num_tokens, old_dim)]
            )
        elif new_num_tokens < old_num_tokens:
            scales = scales[:new_num_tokens]
        self.scales.weight.data = scales

    def resize_original_embeddings(self, new_num_tokens: int) -> None:
        """Resize the original embeddings (main weight) with zeros.

        Args:
            new_num_tokens: Total number of tokens (original + added).
        """
        # Resize the main embedding weight
        old_weight = self.weight.data
        old_num_tokens, old_dim = old_weight.shape

        if new_num_tokens > old_num_tokens:
            # Add new embeddings initialized to zeros
            new_weight = torch.cat(
                [
                    old_weight,
                    old_weight.new_zeros(new_num_tokens - old_num_tokens, old_dim),
                ]
            )
            self.weight.data = new_weight
        elif new_num_tokens < old_num_tokens:
            self.weight.data = old_weight[:new_num_tokens]

    def resize_added_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize the added tokens embeddings.

        Args:
            new_num_tokens: Total number of tokens (original + added).
        """
        new_num_added_tokens = new_num_tokens - self.original_vocab_size

        # Update added token embeddings
        added_embeds = self.added_tokens_embeddings.weight.data
        old_num_tokens, old_dim = added_embeds.shape

        if new_num_added_tokens > old_num_tokens:
            # Add new embeddings (initialized to zeros or you can use a different strategy)
            added_embeds = torch.cat(
                [
                    added_embeds,
                    added_embeds.new_zeros(
                        new_num_added_tokens - old_num_tokens, old_dim
                    ),
                ]
            )
        elif new_num_added_tokens < old_num_tokens:
            added_embeds = added_embeds[:new_num_added_tokens]

        self.added_tokens_embeddings.weight.data = added_embeds

        # Also resize scales if enabled
        if self.use_scale:
            self.resize_token_scales(new_num_tokens)

    def get_added_token_embeddings(self) -> torch.Tensor:
        """Get the embeddings for added tokens only.

        Returns:
            Tensor of shape (num_added_tokens, embedding_dim)
        """
        return self.added_tokens_embeddings.weight.data

    def set_added_tokens_embeddings(self, embeddings: torch.Tensor) -> None:
        """Set the embeddings for added tokens.

        Args:
            embeddings: Tensor of shape (num_added_tokens, embedding_dim)
        """
        self.added_tokens_embeddings.weight.data = embeddings


class DtiTokenEmbeddingLegacy(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        use_scale: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            *args,
            **kwargs,
        )
        self.original_vocab_size = num_embeddings
        self.use_scale = use_scale
        self.scale_cache = {}

        if self.use_scale:
            self.scales = nn.Embedding(0, 1)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        inputs_embeds = super().forward(input_ids)
        index_rescale = input_ids >= self.original_vocab_size
        if self.use_scale and index_rescale.any():
            rescale_ids = self.to_added_token_id(input_ids[index_rescale])
            inputs_scales = self.scales(rescale_ids)
            inputs_embeds[index_rescale] = inputs_scales * inputs_embeds[index_rescale]
        return inputs_embeds

    @torch.no_grad()
    def _compute_embedding_scales(self) -> torch.Tensor:
        """Compute norms of all embeddings, excluding zeros."""
        scales = torch.norm(self.weight, dim=-1, keepdim=True)
        return scales[scales > 0]

    @torch.no_grad()
    def get_scale_stat(self, stat: str = "mean") -> float:
        """Get a statistic of the embedding scales.

        Args:
            stat: Statistic to compute ("mean", "min", "max").

        Returns:
            The computed statistic value.
        """
        if stat not in SCALE_TYPES:
            raise ValueError(f"Unsupported stat: {stat}. Must be one of {SCALE_TYPES}")
        if stat not in self.scale_cache:
            scales = self._compute_embedding_scales()
            self.scale_cache["min"] = scales.min().item()
            self.scale_cache["mean"] = scales.mean().item()
            self.scale_cache["max"] = scales.max().item()
        return self.scale_cache[stat]

    def to_added_token_id(
        self, token_id: int | list[int] | torch.LongTensor
    ) -> int | list[int] | torch.LongTensor:
        """Convert global token ID(s) to added token ID(s).

        Args:
            token_id: Global token ID (int), list of IDs, or tensor of IDs (>= original_vocab_size).

        Returns:
            Added token ID(s) in the same type as input (token_id - original_vocab_size).
        """
        if isinstance(token_id, torch.Tensor):
            return token_id - self.original_vocab_size
        elif isinstance(token_id, list):
            return [tid - self.original_vocab_size for tid in token_id]
        else:
            return token_id - self.original_vocab_size

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        """Resize token embeddings and scales.

        Unified API for V1: resizes main weight (via parent) and scales.

        Args:
            new_num_tokens: New vocabulary size (total tokens).
            pad_to_multiple_of: Optional padding to multiple.
        """
        if pad_to_multiple_of is not None:
            new_num_tokens = (
                (new_num_tokens + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        # Resize main embedding weight
        old_num_embeddings = self.num_embeddings
        old_weight = self.weight.data

        if new_num_tokens != old_num_embeddings:
            self.num_embeddings = new_num_tokens
            if new_num_tokens > old_num_embeddings:
                # Expand with zeros
                self.weight = nn.Parameter(
                    torch.cat(
                        [
                            old_weight,
                            old_weight.new_zeros(
                                new_num_tokens - old_num_embeddings, self.embedding_dim
                            ),
                        ]
                    )
                )
            else:
                self.weight = nn.Parameter(old_weight[:new_num_tokens])

        # Resize scales
        self.resize_token_scales(new_num_tokens)

    def resize_token_scales(self, new_num_tokens: int) -> None:
        if not self.use_scale:
            return

        new_num_tokens = new_num_tokens - self.original_vocab_size

        # Update token scale embeddings.
        # This is needed because the token scale embeddings are not tied to the token.
        scales = self.scales.weight.data
        old_num_tokens, old_dim = scales.shape
        if new_num_tokens is None:
            new_num_tokens = old_num_tokens
        if new_num_tokens > old_num_tokens:
            scales = torch.cat(
                [scales, scales.new_zeros(new_num_tokens - old_num_tokens, old_dim)]
            )
        elif new_num_tokens < old_num_tokens:
            scales = scales[:new_num_tokens]
        self.scales.weight.data = scales


__all__ = ["DtiTokenEmbedding", "DtiTokenEmbeddingLegacy"]
