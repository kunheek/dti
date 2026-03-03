import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def print_closest_token(
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    target_token_id: int | list[int],
) -> None:
    """Print the closest token embedding and corresponding cosine distance.

    Args:
        tokenizer: Tokenizer with decode method
        text_encoder: Text encoder with get_input_embeddings method
        target_token_id: Index of the target token, or a list of token indices
    """
    embeddings = text_encoder.get_input_embeddings()
    token_embeds = embeddings.weight.data

    # Handle single token or sequence of tokens
    if isinstance(target_token_id, int):
        target_token_ids = [target_token_id]
    else:
        target_token_ids = target_token_id

    # Normalize all embeddings once
    all_embeds_norm = token_embeds / torch.norm(token_embeds, dim=-1, keepdim=True)

    # Print closest token for each target token
    for idx, tid in enumerate(target_token_ids):
        # Get target embedding and normalize
        target_embed = token_embeds[tid]
        target_embed_norm = target_embed / torch.norm(
            target_embed, dim=-1, keepdim=True
        )

        # Compute cosine similarities
        cosine_sims = torch.matmul(all_embeds_norm, target_embed_norm)

        # Exclude all target tokens from consideration
        for exclude_tid in target_token_ids:
            cosine_sims[exclude_tid] = -float("inf")

        # Find closest token
        closest_token_id = torch.argmax(cosine_sims).item()
        cosine_distance = 1 - cosine_sims[closest_token_id].item()

        # Decode tokens
        target_token = tokenizer.decode([tid])
        closest_token = tokenizer.decode([closest_token_id])

        if len(target_token_ids) > 1:
            print(f"[{idx}] Target token ID: {tid}, token: '{target_token}'")
            print(
                f"    Closest token ID: {closest_token_id}, token: '{closest_token}', distance: {cosine_distance:.6f}"
            )
        else:
            print(f"Target token ID: {tid}")
            print(f"Target token: '{target_token}'")
            print(f"Closest token ID: {closest_token_id}")
            print(f"Closest token: '{closest_token}'")
            print(f"Cosine distance: {cosine_distance:.6f}")
