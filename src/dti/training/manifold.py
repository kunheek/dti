import torch


@torch.no_grad()
def project_grads_to_tangent_space(
    text_encoder: torch.nn.Module,
    added_token_ids: list[int],
    *,
    kappa: float = 0.0,
    target_embeds: torch.Tensor | None = None,
    t: int = 0,
    v_tm1: torch.Tensor | None = None,
    beta2: float = 0.99,
    eps: float = 1e-6,
) -> None:
    embeds = text_encoder.get_input_embeddings().weight[
        min(added_token_ids) : max(added_token_ids) + 1
    ]
    grad = text_encoder.get_input_embeddings().weight.grad[
        min(added_token_ids) : max(added_token_ids) + 1
    ]

    # (Optional) Apply cosine similarity regularization.
    if kappa > 0.0 and target_embeds is not None:
        prior_grad = kappa * target_embeds  # - \kappa * \mu
        grad = grad - prior_grad

    # Project the gradient onto the tangent space.
    grad_proj = (embeds * grad).sum(dim=-1, keepdim=True)
    grad = grad - grad_proj * embeds

    # Adaptive gradient normalization.
    h = grad.square().sum(dim=1, keepdim=True)
    beta2_t = beta2 * ((1 - beta2**t) / (1 - beta2 ** (t + 1)))
    v_t = beta2_t * v_tm1 + (1 - beta2_t) * h
    grad = grad / torch.sqrt(v_t).clamp_min(eps)

    # NOTE: legacy implementation used in the earlier versions.
    # If beta2_t = 0, it's equivalent to the above implementation.
    # Normalize the gradient to have norm 1.
    # grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True)  # L2 norm
    # grad_norm = torch.clamp(grad_norm, min=eps)
    # grad = grad / grad_norm

    # Write back the projected gradient.
    text_encoder.get_input_embeddings().weight.grad[
        min(added_token_ids) : max(added_token_ids) + 1
    ] = grad
    return v_t


def retract_token_embeddings(
    embeddings: torch.nn.Embedding,
    index_updates: torch.BoolTensor,
) -> None:
    v = embeddings.weight[index_updates].clone()
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    embeddings.weight[index_updates] = v
