import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def str2bool(v: bool | str | int) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "1")
    elif isinstance(v, int):
        return v == 1
    else:
        raise ValueError(
            f"Invalid value for str2bool: {v}. Expected a boolean, string, or integer."
        )


RESCALE_TYPES = {"min", "mean", "max"}


def _compute_vocab_norm_stat(embeddings: torch.Tensor, stat: str) -> float:
    """Compute a norm statistic (min/mean/max) over non-zero vocabulary embeddings."""
    norms = torch.norm(embeddings, dim=-1)
    norms = norms[norms > 0]
    if stat == "min":
        return norms.min().item()
    elif stat == "mean":
        return norms.mean().item()
    elif stat == "max":
        return norms.max().item()
    else:
        raise ValueError(
            f"Unsupported rescale stat: {stat}. Must be one of {RESCALE_TYPES}"
        )


def load_embedding(
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    state_dict: dict | str,
    placeholder: str | None = None,
    joiner: str = "",
    rescale: str | None = None,
) -> list[str]:
    """Load learned token embeddings into a tokenizer / text encoder.

    Args:
        tokenizer: The tokenizer to add new tokens to.
        text_encoder: The text encoder whose embedding layer will be updated.
        state_dict: Path to a safetensors file or an already-loaded state dict.
        placeholder: Optional replacement placeholder token name.
        joiner: String used to join multi-vector token names.
        rescale: If set, rescale loaded embeddings so their norm matches
            the given vocabulary statistic. One of ``"min"``, ``"mean"``,
            ``"max"``, or ``None`` (no rescaling, default).

    Returns:
        List of identifier strings (one per key in the state dict).
    """
    if rescale is not None and rescale not in RESCALE_TYPES:
        raise ValueError(
            f"Invalid rescale value: {rescale!r}. Must be one of {RESCALE_TYPES} or None."
        )

    assert Path(state_dict).exists(), f"File not found: {state_dict}"
    if not isinstance(state_dict, dict):
        state_dict = load_file(state_dict)

    # Load multiple token embeddings.
    identifiers = []
    for key, embs in state_dict.items():
        if placeholder is not None:
            key = placeholder
        tokens = [key]
        for i in range(1, embs.size(0)):
            tokens.append(f"{key}_{i}")
        tokenizer.add_tokens(tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embeddings = text_encoder.get_input_embeddings().weight.data

        # Optionally rescale each loaded embedding to match a vocab norm stat.
        if rescale is not None:
            target_norm = _compute_vocab_norm_stat(embeddings, rescale)
            rescaled = []
            for emb in embs:
                norm = torch.norm(emb).clamp(min=1e-6)
                rescaled.append(emb * (target_norm / norm))
            embs = torch.stack(rescaled)

        for id, emb in zip(token_ids, embs):
            embeddings[id] = emb.clone()
        identifier = joiner.join(tokens)
        identifiers.append(identifier)
    return identifiers


def data_loop(dataloader: torch.utils.data.DataLoader, max_steps: int) -> Any:
    step = 0
    while step < max_steps:
        for batch in dataloader:
            yield step, batch
            step += 1
            if step >= max_steps:
                break


def generate_exponential_samples_in_range(scale, size=100, low=0, high=20):
    """
    Generates samples from an exponential distribution, and then clips them to fall
    within a specified range.

    Parameters:
    scale (float): The scale parameter (beta) of the exponential distribution.
                   This is the inverse of the rate parameter (lambda).
    size (int): The number of samples to generate. Default is 100.
    low (int or float): The minimum value of the range. Default is 1.
    high (int or float): The maximum value of the range. Default is 10.

    Returns:
    numpy.ndarray: Array of samples from the clipped exponential distribution.
                   Values are guaranteed to be within the range [low, high].
    """
    if scale <= 0:
        raise ValueError("Scale parameter (scale) must be positive.")
    if low >= high:
        raise ValueError("low must be strictly less than high.")

    # Generate samples from the exponential distribution.
    samples = np.random.exponential(scale=scale, size=size)

    # Clip the samples to the specified range.
    clipped_samples = np.clip(samples, low, high)
    return np.floor(clipped_samples).astype(int)


def compute_kappa(R_bar, p):
    return (R_bar * p - R_bar**3) / (1 - R_bar**2)


@torch.no_grad()
def get_close_words(
    target,
    tokenizer,
    text_encoder,
    n=20,
    distance="cosine",
):
    if isinstance(target, str):
        token_id = tokenizer.encode(target, add_special_tokens=False)
        if len(token_id) > 1:
            print(token_id)
            return "Only single tokens are supported."
        token_id = token_id[0]
    else:
        token_id = target

    embeds = text_encoder.get_input_embeddings().weight.data

    if distance == "l2":
        embeds /= embeds.norm(dim=-1, keepdim=True)
        l2 = F.pairwise_distance(embeds, embeds[token_id].unsqueeze(0), p=2)
        l2 = l2.cpu().numpy()
        topk = l2.argsort()[1 : n + 1]
    elif distance == "cosine":
        cos_sim = F.cosine_similarity(embeds, embeds[token_id].unsqueeze(0), dim=-1)
        cos_sim = cos_sim.cpu().numpy()
        cos_dist = 1 - cos_sim
        topk = cos_dist.argsort()[1 : n + 1]
    else:
        raise ValueError("Distance metric not supported.")
    return topk, tokenizer.convert_ids_to_tokens(topk)


def estimate_kappa(
    target,
    tokenizer,
    text_encoder,
    n=500,
    as_tensor=False,
):
    token_id = tokenizer.encode(target, add_special_tokens=False)
    token_embedding = (
        text_encoder.get_input_embeddings()
        .weight.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    mus = []
    kappas = []
    for id in token_id:
        close_ids, _ = get_close_words(id, tokenizer, text_encoder, n=n)

        x = token_embedding[close_ids]
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        mean = x.mean(axis=0)
        norm = np.linalg.norm(mean)
        R_bar = norm
        kappa = compute_kappa(R_bar, token_embedding.shape[1])

        mus.append(mean / norm)
        kappas.append(kappa)

    mu = np.asarray(mus)
    if as_tensor:
        mu = torch.as_tensor(mu)
    kappa = np.asarray(kappas)
    if as_tensor:
        kappa = torch.as_tensor(kappa)
    return mu, kappa
