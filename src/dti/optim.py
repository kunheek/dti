from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class SphericalSGD(Optimizer):
    """Stochastic Gradient Descent on the unit sphere with optional cosine regularization.

    Performs Riemannian optimization where parameters are constrained to lie on
    the unit sphere. Gradients are projected to the tangent space before updates,
    and parameters are retracted back to the manifold after each step.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        kappa: Cosine regularization coefficient (default: 0.0)
        target_embeds: Target embeddings for cosine regularization
        beta: Decay rate for gradient second moment estimation (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        kappa: float = 0.0,
        target_embeds: Optional[torch.Tensor] = None,
        target_ids: Optional[list[int]] = None,
        beta: float = 0.0,
        eps: float = 1e-8,
        # For ablation studies.
        project_to_tangent: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= kappa:
            raise ValueError(f"Invalid kappa: {kappa}")

        defaults = dict(
            lr=lr,
            kappa=kappa,
            target_embeds=target_embeds,
            target_ids=target_ids,
            beta=beta,
            eps=eps,
            project_to_tangent=project_to_tangent,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    # Store per-element gradient variance estimate
                    state["v_t"] = torch.zeros(
                        (*p.shape[:-1], 1), device=p.device, dtype=p.dtype
                    )

                state["step"] += 1

                # Get gradient
                grad = p.grad.data

                # Step 1: Cosine regularization - add constant gradient
                if group["kappa"] > 0.0 and group["target_embeds"] is not None:
                    # Gradient term: -kappa * (target - current)
                    # Simplified to: -kappa * target (constant term)
                    target_ids = group["target_ids"]
                    if target_ids is not None:
                        grad_slice = grad[min(target_ids) : max(target_ids) + 1]
                        # print("grad.shape:", grad.shape)
                        # print("target_ids:", target_ids)
                        # print("grad_slice.shape:", grad_slice.shape)
                        grad_slice.sub_(group["target_embeds"], alpha=group["kappa"])
                    else:
                        grad = grad.sub(group["target_embeds"], alpha=group["kappa"])

                # Step 2: Project to tangent space
                # Tangent projection: g_tan = g - <g, p> * p
                inner_product = (grad * p.data).sum(dim=-1, keepdim=True)
                grad = grad - inner_product * p.data

                # Step 3: Adaptive gradient normalization
                h = grad.square().sum(dim=-1, keepdim=True)
                t = state["step"]
                beta = group["beta"]

                # Bias-corrected exponential moving average
                # beta_corrected = beta * (1 - beta**t) / (1 - beta ** (t + 1))
                # state["v_t"].mul_(beta_corrected).add_(h, alpha=1 - beta_corrected)
                state["v_t"].mul_(beta).add_(h, alpha=1 - beta)
                v_t_corrected = state["v_t"] / (1 - beta**t)

                # Normalize gradient magnitude
                # \sqrt{v_t + eps} vs \sqrt{v_t} + eps: \sqrt{v_t} + eps is better.
                # grad = grad / torch.sqrt(state["v_t"] + group["eps"])
                # grad = grad / torch.sqrt(state["v_t"]).add(group["eps"])
                grad = grad / torch.sqrt(v_t_corrected).add(group["eps"])

                # Step 4: Update parameters
                p.data.add_(grad, alpha=-group["lr"])

                # Step 5: Retraction - project back to unit sphere
                norm = torch.linalg.norm(p.data, dim=-1, keepdim=True)
                p.data.div_(norm)

        return loss
