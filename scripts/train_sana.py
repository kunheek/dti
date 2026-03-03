#!/usr/bin/env python3
import argparse
import copy
import random
import shutil
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderDC,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from transformers import AutoTokenizer, Gemma2Model

from dti.argument_parser import parse_sana_args
from dti.datasets import SanaDataset
from dti.optim import SphericalSGD
from dti.training.token_embedding_ops import (
    add_new_token,
    replace_token_embedding,
)
from dti.training_utils import (
    TrainingLogger,
    prepare_for_training,
    register_embedding_only_checkpoint_hooks,
    setup_accelerator,
    setup_logging,
    save_embeddings,
)
from dti.utils import data_loop
from dti.model_configs import SanaConfig

# from dti.training.debug import print_closest_token
# from dti.training.manifold import (
#     project_grads_to_tangent_space,
#     retract_token_embeddings,
# )

# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def log_validation(
    text_encoder: Gemma2Model,
    tokenizer: AutoTokenizer,
    dit: SanaTransformer2DModel,
    vae: AutoencoderDC,
    args: argparse.Namespace,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=dit,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    print(args.validation_prompt)
    for _ in range(args.num_validation_images):
        image = pipeline(
            args.validation_prompt,
            num_inference_steps=20,
            generator=generator,
            complex_human_instruction=None,  # NOTE: None for validation.
        ).images[0]
        images.append(image)

    del pipeline
    free_memory()

    return images


def main():
    args = parse_sana_args()

    accelerator = setup_accelerator(args)
    setup_logging(accelerator)
    prepare_for_training(accelerator, args, seed=args.seed)

    sana_config = SanaConfig(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    models = sana_config.load_models()
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    dit = models["transformer"]
    noise_scheduler = models["scheduler"]
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Replace token embedding layer with Gemma2TokenEmbedding.
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

    # Add padding token.
    print("Adding padding token")
    placeholder = "<zero-pad>"
    _, _ = add_new_token(
        tokenizer,
        text_encoder,
        placeholder,
        num_vectors=1,
        init_scale=0.0,
    )
    print(new_tokens)

    # Freeze all parameters except for the token embeddings in text encoder
    vae.eval().requires_grad_(False)
    dit.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.embed_tokens.added_tokens_embeddings.requires_grad_(True)
    # if args.decompose_scale:
    #     text_encoder.embed_tokens.scales.requires_grad_(False)
    print(text_encoder.embed_tokens.added_tokens_embeddings.weight.requires_grad)

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        transformer=None,
        scheduler=None,
    )

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer.
    with torch.no_grad():
        target_embeds = init_embeds / torch.linalg.norm(
            init_embeds,
            dim=-1,
            keepdim=True,
        )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            # only optimize the embeddings
            [text_encoder.embed_tokens.added_tokens_embeddings.weight],
            # [text_encoder.embed_tokens.weight, text_encoder.scale_tokens.weight],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = SphericalSGD(
            # [text_encoder.embed_tokens.weight],
            [text_encoder.embed_tokens.added_tokens_embeddings.weight],
            lr=args.learning_rate,
            kappa=args.kappa,
            beta=args.beta,
            target_embeds=target_embeds,
            # target_ids=added_token_ids,
            target_ids=text_encoder.embed_tokens.to_added_token_id(added_token_ids),
        )

    args.validation_prompt = args.validation_prompt.format(new_token.identifier)
    # Dataset and DataLoaders creation:
    train_dataset = SanaDataset(
        data_root=args.train_data_dir,
        instance=args.instance,
        size=args.resolution,
        placeholder_token=new_token.identifier,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        flip_p=0.0,  # NOTE: 0.5?
        zero_pad=args.zero_pad,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(
        prompt,
        pipeline,
        clean_caption=False,
        device=accelerator.device,
    ):
        prompt = pipeline._text_preprocessing(prompt, clean_caption=clean_caption)

        max_length = args.max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        # prepare complex human instruction
        if not args.complex_human_instruction:
            max_length_all = max_length
        else:
            chi_prompt = "\n".join(args.complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_length - 2

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds[0][:, select_index]
        prompt_attention_mask = prompt_attention_mask[:, select_index]

        # sequence_lengths = prompt_attention_mask.sum(dim=1)
        # max_sequence_length = sequence_lengths.max()
        # prompt_embeds = prompt_embeds[:, :max_sequence_length]
        # prompt_attention_mask = prompt_attention_mask[:, :max_sequence_length]
        return prompt_embeds, prompt_attention_mask

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )
    register_embedding_only_checkpoint_hooks(
        accelerator,
        embedding_getters=[lambda model: model.embed_tokens],
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet and text_encoder to device and cast to weight_dtype
    dit.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.run_name is not None:
            # add random two digits to the run name to avoid name clashes
            run_name = f"{args.run_name}-{random.randint(0, 99):02d}"
            init_kwargs["wandb"] = {"name": run_name}
        accelerator.init_trackers("dti", config=vars(args), init_kwargs=init_kwargs)

    # Train!
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
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = Path(args.resume_from_checkpoint).name
        else:
            # Get the most recent checkpoint
            dirs = list(Path(args.output_dir).iterdir())
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for step, batch in data_loop(
        train_dataloader, args.max_train_steps * args.gradient_accumulation_steps
    ):
        prompts = batch["prompt"]

        with accelerator.accumulate(text_encoder):
            prompt_embeds, prompt_attention_mask = compute_text_embeddings(
                prompts, text_encoding_pipeline
            )

            # Convert images to latent space
            model_input = vae.encode(
                batch["pixel_values"].to(dtype=vae.dtype)
            ).latent.detach()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
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

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(
                timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
            )
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            # Predict the noise residual
            model_pred = dit(
                hidden_states=noisy_model_input.to(dtype=weight_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                encoder_attention_mask=prompt_attention_mask,
                timestep=timesteps,
                return_dict=False,
            )[0]

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.weighting_scheme, sigmas=sigmas
            )

            # flow matching loss
            target = noise - model_input

            # Compute regular loss.
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            # L2 regularization between current and initial embeddings (CrossInit).
            if args.reg_lambda > 0.0:
                embed_tokens = accelerator.unwrap_model(text_encoder).embed_tokens
                emb = embed_tokens.added_tokens_embeddings.weight
                added_ids = embed_tokens.to_added_token_id(added_token_ids)
                reg_loss = F.mse_loss(emb[added_ids], init_embeds.to(emb.dtype))
                loss = loss + args.reg_lambda * reg_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
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
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = list(Path(args.output_dir).iterdir())
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = (
                                    Path(args.output_dir) / removing_checkpoint
                                )
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
                        args,
                        accelerator,
                        weight_dtype,
                    )
                    rows = args.num_validation_images // 2
                    cols = 2
                    image_grid = diffusers.utils.make_image_grid(
                        images, rows=rows, cols=cols
                    )
                    image_grid.save(
                        Path(args.output_dir) / f"validation-{global_step:04d}.jpg"
                    )

            embed_tokens = accelerator.unwrap_model(text_encoder).embed_tokens
            train_logger.log(
                global_step=global_step,
                loss=loss.detach().item(),
                embeddings_list=[embed_tokens],
                token_ids_list=[added_token_ids],
                lr=args.learning_rate,
            )

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(
                text_encoder,
                tokenizer,
                dit,
                vae,
                args,
                accelerator,
                weight_dtype,
            )

        # Save the newly trained embeddings
        weight_name = "learned_embeds.safetensors"
        save_path = Path(args.output_dir) / weight_name
        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            save_path,
            safe_serialization=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
