import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from ..embedding import DtiTokenEmbeddingLegacy, DtiTokenEmbedding


@dataclass
class NewToken:
    placeholder: str
    identifier: str
    initializer: str
    token_ids: list[int]
    num_vectors: int


def replace_token_embedding(
    model: PreTrainedModel,
    use_scale: bool = True,
    legacy: bool = False,
) -> nn.Module:
    """Replace model's token embedding with DtiTokenEmbedding.

    Args:
        model: Model with get_input_embeddings/set_input_embeddings methods.
        use_scale: Whether to use learnable scale parameters.
        version: Embedding version (1 or 2). V1 stores all embeddings in main weight,
                 V2 separates added tokens into added_tokens_embeddings.

    Returns:
        Model with replaced embedding layer.
    """
    old_embedding = model.get_input_embeddings()
    if isinstance(old_embedding, (DtiTokenEmbeddingLegacy, DtiTokenEmbedding)):
        raise RuntimeError("Model is already using DtiTokenEmbedding.")

    if legacy:
        new_embedding = DtiTokenEmbeddingLegacy(
            old_embedding.num_embeddings,
            old_embedding.embedding_dim,
            padding_idx=old_embedding.padding_idx,
            use_scale=use_scale,
        ).to(old_embedding.weight.device)
    else:
        new_embedding = DtiTokenEmbedding(
            old_embedding.num_embeddings,
            old_embedding.embedding_dim,
            padding_idx=old_embedding.padding_idx,
            use_scale=use_scale,
        ).to(old_embedding.weight.device)

    new_embedding.weight.data[: old_embedding.num_embeddings] = (
        old_embedding.weight.data
    )

    model.set_input_embeddings(new_embedding)

    if not torch.equal(old_embedding.weight, model.get_input_embeddings().weight):
        raise RuntimeError("Embedding weights are not exactly the same after resizing.")

    return model


@torch.no_grad()
def token_cross_init(
    tokens: str | list[str],
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    return_input_embeds: bool = False,
    max_length: int | None = None,
) -> torch.Tensor:
    """CrossInit: initialize new token embeddings using contextual representations.

    Generalizes across tokenizer/text_encoder pairs (CLIP, Gemma2, etc.) by
    finding content token positions via subsequence matching, which correctly
    handles both left-padded (Gemma2) and right-padded (CLIP) sequences.

    Args:
        tokens: Initializer token(s) as a string or list of strings.
        tokenizer: Any HuggingFace tokenizer.
        text_encoder: Any HuggingFace text encoder model.
        return_input_embeds: If True, return static input embeddings instead of
            contextual (last hidden state) embeddings.
        max_length: Maximum sequence length for tokenization. If None, uses
            ``tokenizer.model_max_length``.

    Returns:
        Tensor of shape ``(num_content_tokens, hidden_dim)`` containing
        embeddings for the initializer tokens only.
    """
    if isinstance(tokens, list):
        tokens = " ".join(tokens)

    if max_length is None:
        model_limit = None
        config = getattr(text_encoder, "config", None)
        if config is not None:
            for attr in (
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
            ):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    model_limit = value
                    break

        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
            max_length = tokenizer_limit
        else:
            max_length = model_limit

        # Some tokenizer configs use an "infinite" sentinel such as
        # 1000000000000000019884624838656; this overflows fast-tokenizer truncation.
        if not isinstance(max_length, int) or max_length > 1_000_000:
            max_length = model_limit

        if not isinstance(max_length, int) or max_length <= 0:
            max_length = 512

    max_length = int(max_length)

    # Tokenize with full padding to match training-time layout.
    encoded = tokenizer(
        tokens,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids.to(text_encoder.device)

    # Get the content token IDs (without any special tokens like BOS/EOS/PAD).
    content_ids = tokenizer.encode(tokens, add_special_tokens=False)

    # Find the content token subsequence inside the full padded sequence.
    token_ids_list = token_ids[0].tolist()
    content_positions = _find_subsequence(token_ids_list, content_ids)
    if content_positions is None:
        raise ValueError(
            f"Could not locate content tokens {content_ids} in the "
            f"encoded sequence {token_ids_list}."
        )

    # Get embeddings.
    if return_input_embeds:
        embeds = text_encoder.get_input_embeddings()(token_ids)[0]  # (seq_len, d)
    else:
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(text_encoder.device)
        outputs = text_encoder(
            token_ids,
            attention_mask=attention_mask,
        )
        embeds = outputs.last_hidden_state[0]  # (seq_len, d)

    return embeds[content_positions]


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> list[int] | None:
    """Find the first occurrence of *subsequence* in *sequence*.

    Returns:
        List of indices into *sequence*, or ``None`` if not found.
    """
    n = len(subsequence)
    for start in range(len(sequence) - n + 1):
        if sequence[start : start + n] == subsequence:
            return list(range(start, start + n))
    return None


@torch.no_grad()
def add_new_token(
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    placeholder: str,
    num_vectors: int,
    init_token: torch.Tensor | str | None = None,
    init_method: str = "random",
    init_scale: str = "mean",
    joiner: str = "",
    xinit_max_length: int | None = None,
) -> tuple[NewToken, torch.Tensor]:
    tokenizer_base_size = len(tokenizer)

    if isinstance(init_token, str):
        init_token_ids = tokenizer.encode(init_token, add_special_tokens=False)
        num_vectors = len(init_token_ids)
    elif isinstance(init_token, torch.Tensor):
        init_token_ids = None
        num_vectors = init_token.shape[0]

    placeholder_tokens = [placeholder]
    additional_tokens = []
    for i in range(1, num_vectors):
        additional_tokens.append(f"{placeholder}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains one of the tokens {placeholder_tokens}. Please pass a different `placeholder`."
        )
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    embeddings = text_encoder.get_input_embeddings()

    # Some tokenizers (e.g., certain LLM tokenizers) can report a vocab size that
    # is smaller than the embedding table size. In that case, DTI's boundary for
    # "added tokens" must follow tokenizer IDs, otherwise added-token offsets can
    # become negative.
    if isinstance(embeddings, (DtiTokenEmbeddingLegacy, DtiTokenEmbedding)):
        if (
            embeddings.original_vocab_size != tokenizer_base_size
            and getattr(embeddings, "added_tokens_embeddings", None) is not None
            and embeddings.added_tokens_embeddings.weight.shape[0] == 0
        ):
            warnings.warn(
                "Tokenizer size and embedding original_vocab_size differ "
                f"({tokenizer_base_size} vs {embeddings.original_vocab_size}). "
                "Aligning DTI added-token boundary to tokenizer size.",
                UserWarning,
                stacklevel=2,
            )
            embeddings.original_vocab_size = tokenizer_base_size
        elif (
            embeddings.original_vocab_size != tokenizer_base_size
            and isinstance(embeddings, DtiTokenEmbeddingLegacy)
            and embeddings.scales.weight.shape[0] == 0
        ):
            warnings.warn(
                "Tokenizer size and embedding original_vocab_size differ "
                f"({tokenizer_base_size} vs {embeddings.original_vocab_size}). "
                "Aligning DTI added-token boundary to tokenizer size.",
                UserWarning,
                stacklevel=2,
            )
            embeddings.original_vocab_size = tokenizer_base_size

    if isinstance(embeddings, (DtiTokenEmbeddingLegacy, DtiTokenEmbedding)):
        # Use unified resize API - handles original, added tokens, and scales automatically
        embeddings.resize_token_embeddings(len(tokenizer))
    else:
        # For standard embeddings, use text_encoder's resize method
        text_encoder.resize_token_embeddings(len(tokenizer))
        embeddings = text_encoder.get_input_embeddings()

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = embeddings.weight.data

    # Warn early if xinit scale override is attempted (before conversion).
    if init_method == "xinit" and init_scale != "mean":
        warnings.warn(
            f"init_method='xinit' uses the norm of contextual embeddings as the "
            f"scale. The provided init_scale={init_scale!r} will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    has_scale = hasattr(embeddings, "scales") and embeddings.use_scale
    if init_method == "xinit":
        # CrossInit stores the full contextual embedding (direction + norm combined),
        # so the separate scale component is not used.
        has_scale = False
    if has_scale:
        token_scales = embeddings.scales.weight.data
        if isinstance(init_scale, str):
            init_scale = embeddings.get_scale_stat(init_scale)
        else:
            init_scale = float(init_scale)

    # For CrossInit, compute all contextual embeddings up front.
    xinit_embeds = None
    if init_method == "xinit":
        if not isinstance(init_token, str):
            raise ValueError(
                "init_method='xinit' requires init_token to be a string "
                "(the initializer word/phrase)."
            )
        xinit_embeds = token_cross_init(
            init_token,
            tokenizer,
            text_encoder,
            max_length=xinit_max_length,
        )  # (num_content_tokens, d)
        xinit_scales = xinit_embeds.norm(dim=-1)  # (num_content_tokens,)
        print(
            "[CrossInit] contextual embedding norms (used as scale): "
            + ", ".join(f"{s:.4f}" for s in xinit_scales.tolist())
        )

    init_embeds = []
    for i in range(num_vectors):
        token_id = placeholder_token_ids[i]
        if init_method == "xinit":
            init_embed = xinit_embeds[i].clone()
        elif init_token is None:
            init_embed = torch.randn_like(token_embeds[0])
        elif init_token_ids is not None:
            init_token_id = init_token_ids[i]
            init_embed = token_embeds[init_token_id].clone()
        else:
            init_embed = init_token[i].clone()
        init_embeds.append(init_embed)

        if init_method == "token":
            new_embed = init_embed.clone()
        elif init_method == "random":
            new_embed = torch.randn_like(init_embed)
        elif init_method == "mean":
            valid_embeds = token_embeds[: embeddings.original_vocab_size]
            # remove zero embeddings
            valid_embeds = valid_embeds[torch.norm(valid_embeds, dim=-1) > 0.0]
            new_embed = torch.mean(valid_embeds, dim=0)
        elif init_method == "xinit":
            new_embed = init_embed.clone()
        else:
            raise ValueError(f"Unknown init method: {init_method}.")
        if has_scale:  # Scale is controlled separately
            new_embed = new_embed / torch.norm(new_embed, dim=-1, keepdim=True).clamp(
                min=1e-6
            )

        # Write the new embedding to the appropriate location
        if isinstance(embeddings, DtiTokenEmbedding):
            embeddings.write_token_embedding(token_id, new_embed)
        else:
            # For V1 or standard embeddings, write to main token_embeds
            token_embeds[token_id] = new_embed

        if has_scale:
            scale_id = embeddings.to_added_token_id(token_id)
            token_scales[scale_id] = init_scale

    init_embeds = torch.stack(init_embeds, dim=0)
    new_token = NewToken(
        placeholder=placeholder,
        identifier=joiner.join(placeholder_tokens),
        initializer=init_token if isinstance(init_token, str) else "",
        token_ids=placeholder_token_ids,
        num_vectors=num_vectors,
    )
    return new_token, init_embeds
