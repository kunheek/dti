#!/usr/bin/env python3
import argparse
import copy
import random
import shutil
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from PIL import Image
from transformers import UMT5EncoderModel

from dti.argument_parser import parse_wan_args
from dti.datasets import SanaDataset
from dti.model_configs import WanConfig
from dti.optim import SphericalSGD
from dti.training.token_embedding_ops import add_new_token, replace_token_embedding
from dti.training_utils import (
    TrainingLogger,
    prepare_for_training,
    register_embedding_only_checkpoint_hooks,
    save_embeddings,
    setup_accelerator,
    setup_logging,
)
from dti.utils import data_loop

logger = get_logger(__name__)


def build_wan_pipeline(
    args: argparse.Namespace,
    tokenizer=None,
    text_encoder=None,
    transformer=None,
    vae=None,
    scheduler=None,
):
    kwargs = {
        "revision": args.revision,
        "variant": args.variant,
    }
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if text_encoder is not None:
        kwargs["text_encoder"] = text_encoder
    if transformer is not None:
        kwargs["transformer"] = transformer
    if vae is not None:
        kwargs["vae"] = vae
    if scheduler is not None:
        kwargs["scheduler"] = scheduler

    return WanPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        **kwargs,
    )


def first_frame_to_pil(video_frames):
    """Convert a single generated video output into a PIL first frame."""
    frame0 = video_frames[0]
    if isinstance(frame0, Image.Image):
        return frame0
    if torch.is_tensor(frame0):
        arr = frame0.detach().cpu().numpy()
    else:
        arr = np.asarray(frame0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def log_validation(
    text_encoder: UMT5EncoderModel,
    tokenizer,
    dit,
    vae,
    scheduler,
    args: argparse.Namespace,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} videos (using first frame) with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = build_wan_pipeline(
        args,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(text_encoder),
        transformer=dit,
        vae=vae,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.vae.to(dtype=weight_dtype)

    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    for _ in range(args.num_validation_images):
        output = pipeline(
            prompt=args.validation_prompt,
            negative_prompt=args.negative_prompt,
            height=args.resolution,
            width=args.resolution,
            num_frames=args.num_frames,
            num_inference_steps=20,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pil",
        )
        images.append(first_frame_to_pil(output.frames[0]))

    del pipeline
    free_memory()
    return images


def encode_prompt(
    pipeline: WanPipeline,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    weight_dtype: torch.dtype,
):
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompts,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
        device=device,
        dtype=weight_dtype,
    )
    return prompt_embeds


def encode_videos_to_latents(
    vae,
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    videos = pixel_values.unsqueeze(2).to(device=device, dtype=vae.dtype)
    latent_condition = retrieve_latents(vae.encode(videos), sample_mode="argmax").to(
        device=device, dtype=dtype
    )

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device=latent_condition.device, dtype=latent_condition.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(device=latent_condition.device, dtype=latent_condition.dtype)
    return (latent_condition - latents_mean) * latents_std


def main():
    args = parse_wan_args()

    accelerator = setup_accelerator(args)
    setup_logging(accelerator)
    prepare_for_training(accelerator, args, seed=args.seed)

    wan_config = WanConfig(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    models = wan_config.load_models()
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    dit = models["transformer"]
    noise_scheduler = models["scheduler"]
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder = replace_token_embedding(
        text_encoder,
        use_scale=args.decompose_scale,
        legacy=False,
    )

    added_token_ids = []
    new_tokens = []
    new_token, init_embeds = add_new_token(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        num_vectors=args.num_vectors,
        init_token=args.initializer_token,
        init_method=args.init_method,
        init_scale=args.token_scale,
        joiner="",
        xinit_max_length=args.max_sequence_length,
    )
    added_token_ids += new_token.token_ids.copy()
    new_tokens.append(new_token)
    init_embeds = init_embeds.to(accelerator.device)

    # Freeze all parameters except token embeddings.
    vae.eval().requires_grad_(False)
    dit.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)

    token_embeddings = text_encoder.get_input_embeddings()
    if not hasattr(token_embeddings, "added_tokens_embeddings"):
        raise TypeError(
            "Wan training expects DTI embeddings with `added_tokens_embeddings`."
        )
    token_embeddings.added_tokens_embeddings.requires_grad_(True)

    text_encoding_pipeline = build_wan_pipeline(
        args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        vae=vae,
        scheduler=noise_scheduler_copy,
    )
    text_encoding_pipeline.set_progress_bar_config(disable=True)

    if args.gradient_checkpointing and hasattr(
        text_encoder, "gradient_checkpointing_enable"
    ):
        text_encoder.gradient_checkpointing_enable()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        base_lr = args.learning_rate
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        logger.info(
            "Scaled learning rate: %.6g -> %.6g (accum=%d, batch=%d, processes=%d)",
            base_lr,
            args.learning_rate,
            args.gradient_accumulation_steps,
            args.train_batch_size,
            accelerator.num_processes,
        )

    with torch.no_grad():
        target_embeds = init_embeds / torch.linalg.norm(
            init_embeds,
            dim=-1,
            keepdim=True,
        )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [token_embeddings.added_tokens_embeddings.weight],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = SphericalSGD(
            [token_embeddings.added_tokens_embeddings.weight],
            lr=args.learning_rate,
            kappa=args.kappa,
            beta=args.beta,
            target_embeds=target_embeds,
            target_ids=token_embeddings.to_added_token_id(added_token_ids),
        )

    if args.validation_prompt is not None:
        args.validation_prompt = args.validation_prompt.format(new_token.identifier)

    train_dataset = SanaDataset(
        data_root=args.train_data_dir,
        instance=args.instance,
        size=args.resolution,
        placeholder_token=new_token.identifier,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        flip_p=0.0,
        zero_pad=args.zero_pad,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    text_encoder.train()
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )
    text_encoding_pipeline.text_encoder = text_encoder
    register_embedding_only_checkpoint_hooks(
        accelerator,
        embedding_getters=[lambda model: model.get_input_embeddings()],
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    dit.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoding_pipeline.to(accelerator.device)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.run_name is not None:
            run_name = f"{args.run_name}-{random.randint(0, 99):02d}"
            init_kwargs["wandb"] = {"name": run_name}
        accelerator.init_trackers("dti", config=vars(args), init_kwargs=init_kwargs)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = Path(args.resume_from_checkpoint).name
        else:
            dirs = list(Path(args.output_dir).iterdir())
            dirs = [d for d in dirs if d.name.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.name.split("-")[1]))
            path = dirs[-1].name if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(Path(args.output_dir) / path)
            global_step = int(path.split("-")[1])

    train_logger = TrainingLogger(args.output_dir, log_freq=args.log_freq)

    def get_sigmas(timesteps, n_dim=5, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(
            device=accelerator.device
        )
        timesteps = timesteps.to(accelerator.device)
        step_indices = torch.stack(
            [(schedule_timesteps - t).abs().argmin() for t in timesteps]
        ).long()

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for _, batch in data_loop(
        train_dataloader,
        args.max_train_steps * args.gradient_accumulation_steps,
    ):
        prompts = batch["prompt"]

        with accelerator.accumulate(text_encoder):
            prompt_embeds = encode_prompt(
                text_encoding_pipeline,
                prompts,
                args,
                accelerator.device,
                weight_dtype,
            )

            model_input = encode_videos_to_latents(
                vae,
                batch["pixel_values"],
                accelerator.device,
                dtype=weight_dtype,
            ).detach()

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(
                device=model_input.device
            )

            sigmas = get_sigmas(
                timesteps,
                n_dim=model_input.ndim,
                dtype=model_input.dtype,
            )
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            model_pred = dit(
                hidden_states=noisy_model_input.to(dtype=weight_dtype),
                timestep=timesteps.to(device=model_input.device, dtype=weight_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                return_dict=False,
            )[0]

            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.weighting_scheme,
                sigmas=sigmas,
            )
            target = noise - model_input
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            ).mean()

            if args.reg_lambda > 0.0:
                embed_tokens = accelerator.unwrap_model(
                    text_encoder
                ).get_input_embeddings()
                emb = embed_tokens.added_tokens_embeddings.weight
                added_ids = embed_tokens.to_added_token_id(added_token_ids)
                reg_loss = F.mse_loss(emb[added_ids], init_embeds.to(emb.dtype))
                loss = loss + args.reg_lambda * reg_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            images = []
            global_step += 1
            if global_step % args.save_steps == 0:
                weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                save_path = Path(args.output_dir) / weight_name
                save_embeddings(
                    accelerator.unwrap_model(text_encoder),
                    new_tokens,
                    save_path,
                    safe_serialization=True,
                )

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = list(Path(args.output_dir).iterdir())
                        checkpoints = [
                            d for d in checkpoints if d.name.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.name.split("-")[1])
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            for removing_checkpoint in checkpoints[:num_to_remove]:
                                shutil.rmtree(removing_checkpoint)

                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if (
                    args.validation_prompt is not None
                    and global_step % args.validation_steps == 0
                ):
                    images = log_validation(
                        text_encoder,
                        tokenizer,
                        dit,
                        vae,
                        noise_scheduler,
                        args,
                        accelerator,
                        weight_dtype,
                    )
                    rows = max(1, args.num_validation_images // 2)
                    cols = 2 if args.num_validation_images > 1 else 1
                    image_grid = diffusers.utils.make_image_grid(
                        images,
                        rows=rows,
                        cols=cols,
                    )
                    image_grid.save(
                        Path(args.output_dir) / f"validation-{global_step:04d}.jpg"
                    )

            embed_tokens = accelerator.unwrap_model(text_encoder).get_input_embeddings()
            current_lr = optimizer.param_groups[0]["lr"]
            train_logger.log(
                global_step=global_step,
                loss=loss.detach().item(),
                embeddings_list=[embed_tokens],
                token_ids_list=[added_token_ids],
                lr=current_lr,
            )

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            log_validation(
                text_encoder,
                tokenizer,
                dit,
                vae,
                noise_scheduler,
                args,
                accelerator,
                weight_dtype,
            )

        save_path = Path(args.output_dir) / "learned_embeds.safetensors"
        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            save_path,
            safe_serialization=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
