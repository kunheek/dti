"""Common utilities for training scripts."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import SanaPipeline, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import PeftModel
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from .embedding import DtiTokenEmbedding, DtiTokenEmbeddingLegacy
from .training.token_embedding_ops import NewToken


def setup_accelerator(args):
    """Initialize accelerator with common settings."""
    logging_dir = Path(args.output_dir) / args.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator


def setup_logging(accelerator):
    """Configure logging for all processes."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def prepare_for_training(accelerator, args, seed=None):
    """Common preparation steps."""
    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process and args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def save_embeddings(
    text_encoder,
    new_tokens: list[NewToken],
    save_path: str | Path,
    safe_serialization: bool = True,
    clamp_min: float = 1e-6,
) -> None:
    """Save learned token embeddings to file.

    Args:
        text_encoder: Text encoder containing the embeddings.
        new_tokens: List of NewToken objects to save.
        save_path: Path to save the embeddings.
        safe_serialization: Whether to use safetensors format.

    Note:
        Automatically detects whether embeddings are V1 or V2 and handles them appropriately.
    """
    learned_embeds_dict = {}
    for new_token in new_tokens:
        placeholder = new_token.placeholder
        embeddings = text_encoder.get_input_embeddings()

        # Automatically detect version based on embedding type
        if isinstance(embeddings, DtiTokenEmbedding):
            # V2: Use the helper method
            learned_embeds = embeddings.get_learned_embeddings(new_token.token_ids)
        else:
            # V1 or standard embedding
            learned_embeds = (
                embeddings.weight[
                    min(new_token.token_ids) : max(new_token.token_ids) + 1
                ]
                .detach()
                .float()
            )
            if isinstance(embeddings, DtiTokenEmbeddingLegacy) and embeddings.use_scale:
                scale_ids = embeddings.to_added_token_id(new_token.token_ids)
                learned_scales = embeddings.scales.weight[scale_ids].detach().float()
                norms = torch.norm(learned_embeds, dim=-1, keepdim=True).clamp(
                    min=clamp_min
                )
                learned_embeds = learned_scales * (learned_embeds / norms)

        learned_embeds_dict[placeholder] = learned_embeds

    if safe_serialization:
        save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def save_sdxl_peft(unet, save_dir: str):
    if isinstance(unet, PeftModel):
        unet.save_pretrained(save_dir)
        return

    if hasattr(unet, "peft_config"):
        # Save adapters in PEFT format
        state_dict = get_peft_model_state_dict(unet)
        unet.peft_config["default"].save_pretrained(save_dir)
        save_file(state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        return

    lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_dir,
        unet_lora_layers=lora_layers,
    )


def save_sana_peft(transformer, save_dir):
    lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(transformer)
    )
    SanaPipeline.save_lora_weights(
        save_dir,
        transformer_lora_layers=lora_layers,
    )


def _extract_embedding_state(embedding_module) -> dict[str, torch.Tensor]:
    if not hasattr(embedding_module, "added_tokens_embeddings"):
        raise TypeError(
            "Embedding-only checkpointing expects a module with "
            "`added_tokens_embeddings`."
        )

    state = {
        "added_tokens_embeddings.weight": (
            embedding_module.added_tokens_embeddings.weight.detach().cpu()
        ),
    }
    if hasattr(embedding_module, "scales"):
        state["scales.weight"] = embedding_module.scales.weight.detach().cpu()
    return state


def _restore_embedding_state(
    embedding_module,
    state: dict[str, torch.Tensor],
    model_index: int,
) -> None:
    key = "added_tokens_embeddings.weight"
    if key not in state:
        raise KeyError(
            f"Missing key '{key}' in embedding checkpoint for model_{model_index}."
        )

    target = embedding_module.added_tokens_embeddings.weight
    source = state[key]
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for model_{model_index}:{key} - "
            f"checkpoint {tuple(source.shape)} vs model {tuple(target.shape)}."
        )
    target.data.copy_(source.to(device=target.device, dtype=target.dtype))

    scale_key = "scales.weight"
    if scale_key in state:
        if not hasattr(embedding_module, "scales"):
            raise ValueError(
                f"Checkpoint has '{scale_key}' for model_{model_index}, "
                "but current embedding module has no scales."
            )
        scale_target = embedding_module.scales.weight
        scale_source = state[scale_key]
        if scale_target.shape != scale_source.shape:
            raise ValueError(
                f"Shape mismatch for model_{model_index}:{scale_key} - "
                f"checkpoint {tuple(scale_source.shape)} vs model {tuple(scale_target.shape)}."
            )
        scale_target.data.copy_(
            scale_source.to(device=scale_target.device, dtype=scale_target.dtype)
        )


def register_embedding_only_checkpoint_hooks(
    accelerator: Accelerator,
    embedding_getters: list[Callable[[torch.nn.Module], torch.nn.Module]],
    checkpoint_name: str = "embedding_state.bin",
) -> None:
    """Override accelerator state checkpoints to save/load only embedding modules.

    This keeps optimizer/RNG states from ``accelerator.save_state`` while replacing
    full model state dict serialization with compact embedding-only tensors.

    The order of ``embedding_getters`` must match the order of models passed to
    ``accelerator.prepare``.
    """
    if not embedding_getters:
        raise ValueError("`embedding_getters` must contain at least one getter.")

    def save_model_hook(models, weights, output_dir):
        if len(models) < len(embedding_getters):
            raise RuntimeError(
                "Not enough prepared models to save embedding-only checkpoint. "
                f"Expected >= {len(embedding_getters)}, got {len(models)}."
            )

        payload = {}
        for idx, getter in enumerate(embedding_getters):
            model = accelerator.unwrap_model(models[idx])
            embedding_module = getter(model)
            payload[f"model_{idx}"] = _extract_embedding_state(embedding_module)

        torch.save(payload, Path(output_dir) / checkpoint_name)
        del weights[: len(embedding_getters)]

    def load_model_hook(models, input_dir):
        checkpoint_path = Path(input_dir) / checkpoint_name
        if not checkpoint_path.exists():
            # Backward compatibility for old full-model accelerator checkpoints.
            return

        if len(models) < len(embedding_getters):
            raise RuntimeError(
                "Not enough prepared models to load embedding-only checkpoint. "
                f"Expected >= {len(embedding_getters)}, got {len(models)}."
            )

        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(
                f"Invalid embedding checkpoint format at {checkpoint_path}: "
                f"expected dict, got {type(payload)}."
            )

        for idx, getter in enumerate(embedding_getters):
            state = payload.get(f"model_{idx}")
            if state is None:
                raise KeyError(
                    f"Missing entry 'model_{idx}' in embedding checkpoint "
                    f"{checkpoint_path}."
                )
            model = accelerator.unwrap_model(models[idx])
            embedding_module = getter(model)
            _restore_embedding_state(embedding_module, state, idx)

        del models[: len(embedding_getters)]

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


class TrainingLogger:
    """Simple file-based training logger.

    Logs metrics every ``log_freq`` global steps to a JSONL file in the
    output directory and prints a formatted summary to stdout.
    """

    def __init__(self, output_dir: str | Path, log_freq: int = 10) -> None:
        self.output_dir = Path(output_dir)
        self.log_freq = log_freq
        self.log_path = self.output_dir / "train_log.jsonl"
        # Ensure the directory exists (may already).
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any previous log in the same directory.
        if self.log_path.exists():
            self.log_path.unlink()

    @torch.no_grad()
    def _embedding_magnitudes(
        self, embeddings: DtiTokenEmbedding, token_ids: list[int]
    ) -> tuple[float, float]:
        """Return (raw_magnitude, actual_magnitude) for the added token embeddings.

        - raw_magnitude:    mean L2-norm of the raw direction vectors in
                            ``added_tokens_embeddings`` (before scale is applied).
        - actual_magnitude: mean L2-norm of the final saved embeddings
                            (= mean scale value when ``use_scale=True``,
                             same as raw norm when ``use_scale=False``).
        """
        added_ids = embeddings.to_added_token_id(token_ids)
        raw_embeds = embeddings.added_tokens_embeddings.weight[added_ids].float()
        raw_mag = torch.norm(raw_embeds, dim=-1).mean().item()

        if isinstance(embeddings, DtiTokenEmbedding) and embeddings.use_scale:
            # actual = scale (the saved vector is scale * unit_direction)
            actual_mag = embeddings.scales.weight[added_ids].float().mean().item()
        else:
            actual_mag = raw_mag

        return raw_mag, actual_mag

    def log(
        self,
        global_step: int,
        loss: float,
        embeddings_list: list[DtiTokenEmbedding],
        token_ids_list: list[list[int]],
        lr: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Log metrics if ``global_step`` is a multiple of ``log_freq``.

        Args:
            global_step: Current training step.
            loss: Current loss value.
            embeddings_list: List of embedding modules (one per text encoder).
            token_ids_list: Matching list of added-token ID lists.
            lr: Learning rate (optional).
            extra: Any extra key-value pairs to log.
        """
        if global_step % self.log_freq != 0:
            return

        raw_mags, actual_mags = [], []
        for emb, tids in zip(embeddings_list, token_ids_list):
            r, a = self._embedding_magnitudes(emb, tids)
            raw_mags.append(r)
            actual_mags.append(a)
        mean_raw = sum(raw_mags) / len(raw_mags)
        mean_actual = sum(actual_mags) / len(actual_mags)

        record: dict = {
            "step": global_step,
            "loss": round(loss, 6),
            "raw_magnitude": round(mean_raw, 6),
            "actual_magnitude": round(mean_actual, 6),
        }
        if lr is not None:
            record["lr"] = lr
        if extra:
            record.update(extra)

        # Append to JSONL.
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Print a compact summary.
        parts = [f"step={global_step}"]
        parts.append(f"loss={loss:.4f}")
        parts.append(f"raw_mag={mean_raw:.4f}")
        parts.append(f"actual_mag={mean_actual:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")
        print(" | ".join(parts), flush=True)
