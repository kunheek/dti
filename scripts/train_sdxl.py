#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import free_memory
from peft import LoraConfig, OFTConfig
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from dti.argument_parser import parse_sdxl_args
from dti.datasets import SdxlDataset
from dti.model_configs import SDXLConfig
from dti.optim import SphericalSGD
from dti.training.token_embedding_ops import (
    add_new_token,
    replace_token_embedding,
)
from dti.training_utils import (
    TrainingLogger,
    prepare_for_training,
    setup_accelerator,
    setup_logging,
    save_embeddings,
    save_sdxl_peft,
)
from dti.utils import data_loop

logger = get_logger(__name__)


def log_validation(
    accelerator: Accelerator,
    args: argparse.Namespace,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
):
    logger.info(
        "Running validation... \n "
        f"Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder_1),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    logger.info(f"Validation prompt: {args.validation_prompt}")
    for _ in range(args.num_validation_images):
        with accelerator.autocast():
            image = pipeline(
                args.validation_prompt,
                num_inference_steps=25,
                generator=generator,
            ).images[0]
        images.append(image)

    del pipeline
    free_memory()
    return images


def main():
    args = parse_sdxl_args()

    accelerator = setup_accelerator(args)
    setup_logging(accelerator)
    prepare_for_training(accelerator, args, seed=args.seed)

    # Load tokenizers.
    model_config = SDXLConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    models = model_config.load_models()
    tokenizer_1 = models["tokenizer_1"]
    tokenizer_2 = models["tokenizer_2"]
    text_encoder_1 = models["text_encoder_1"]
    text_encoder_2 = models["text_encoder_2"]
    vae = models["vae"]
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]

    # Replace text embeddings with DtiTokenEmbeddings.
    for key in ("text_encoder_1", "text_encoder_2"):
        models[key] = replace_token_embedding(
            models[key],
            use_scale=args.decompose_scale,
            legacy=False,
        )

    added_token_ids = []
    added_token_ids_2 = []
    new_tokens = []
    new_tokens_2 = []

    new_token, init_embeds = add_new_token(
        tokenizer_1,
        text_encoder_1,
        args.placeholder_token,
        num_vectors=args.num_vectors,
        init_token=args.initializer_token,
        init_method=args.init_method,
        init_scale=args.token_scale,
    )
    new_token_2, init_embeds_2 = add_new_token(
        tokenizer_2,
        text_encoder_2,
        args.placeholder_token,
        num_vectors=args.num_vectors,
        init_token=args.initializer_token,
        init_method=args.init_method,
        init_scale=args.token_scale,
    )

    added_token_ids += new_token.token_ids.copy()
    added_token_ids_2 += new_token_2.token_ids.copy()

    new_tokens.append(new_token)
    new_tokens_2.append(new_token_2)

    init_embeds = init_embeds.to(accelerator.device)
    init_embeds_2 = init_embeds_2.to(accelerator.device)

    # Add padding token.
    placeholder = "<pad>"
    _, _ = add_new_token(
        tokenizer_1,
        text_encoder_1,
        placeholder,
        num_vectors=1,
        init_scale=0.0,
    )
    _, _ = add_new_token(
        tokenizer_2,
        text_encoder_2,
        placeholder,
        num_vectors=1,
        init_scale=0.0,
    )
    print(tokenizer_1)
    print(tokenizer_2)
    print(new_tokens)
    print(new_tokens_2)

    # Freeze all parameters.
    vae.eval().requires_grad_(False)
    unet.eval().requires_grad_(False)
    text_encoder_1.eval().requires_grad_(False)
    text_encoder_2.eval().requires_grad_(False)
    # Unfreeze the token embeddings (and scales if using magnitude training).
    # text_encoder_1.get_input_embeddings().requires_grad_(True)
    # text_encoder_2.get_input_embeddings().requires_grad_(True)
    embeddings_1 = text_encoder_1.get_input_embeddings()
    embeddings_2 = text_encoder_2.get_input_embeddings()
    embeddings_1.added_tokens_embeddings.requires_grad_(True)
    embeddings_2.added_tokens_embeddings.requires_grad_(True)
    if args.decompose_scale:
        scale_grad = args.train_magnitude
        embeddings_1.scales.requires_grad_(scale_grad)
        embeddings_2.scales.requires_grad_(scale_grad)

    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()

    if args.use_oft:
        peft_config = OFTConfig(
            r=args.oft_r,
            oft_block_size=0,
            target_modules=args.peft_modules,
            module_dropout=0.0,
            init_weights=True,
        )
    elif args.peft_rank > 0:
        peft_config = LoraConfig(
            r=args.peft_rank,
            lora_alpha=args.peft_rank,
            init_lora_weights="gaussian",
            target_modules=args.peft_modules,
        )

    if args.use_oft or args.peft_rank > 0:
        unet.add_adapter(peft_config)
        # If peft_start_step >= 0, we'll start with frozen LoRA parameters
        if args.peft_start_step >= 0:
            for name, param in unet.named_parameters():
                if ("oft" in name and args.use_oft) or (
                    "lora" in name and not args.use_oft
                ):
                    print(name)
                    param.requires_grad_(False)

    with torch.no_grad():
        target_embeds = init_embeds / torch.linalg.norm(
            init_embeds, dim=-1, keepdim=True
        )
        target_embeds_2 = init_embeds_2 / torch.linalg.norm(
            init_embeds_2, dim=-1, keepdim=True
        )

    # Initialize the optimizer.
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.optimizer == "adamw":
        params_to_train = [
            embeddings_1.added_tokens_embeddings.weight,
            embeddings_2.added_tokens_embeddings.weight,
        ]
        if args.train_magnitude:
            scale_params = [
                embeddings_1.scales.weight,
                embeddings_2.scales.weight,
            ]
            params_to_train = [
                {"params": params_to_train},
                {"params": scale_params, "lr": args.learning_rate},
            ]
        optimizer = torch.optim.AdamW(
            params_to_train,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        # NOTE: make sure target_ids match the ACTUAL added token ids.
        params_to_train = [
            {
                "params": embeddings_1.added_tokens_embeddings.weight,
                "target_embeds": target_embeds,
                "target_ids": embeddings_1.to_added_token_id(added_token_ids),
            },
            {
                "params": embeddings_2.added_tokens_embeddings.weight,
                "target_embeds": target_embeds_2,
                "target_ids": embeddings_2.to_added_token_id(added_token_ids_2),
            },
        ]
        if args.train_magnitude:
            scale_params = [
                embeddings_1.scales.weight,
                embeddings_2.scales.weight,
            ]
            params_to_train.append({"params": scale_params, "target_embeds": None})
        optimizer = SphericalSGD(
            params_to_train,
            lr=args.learning_rate,
            kappa=args.kappa,
            beta=args.beta,
        )

    args.validation_prompt = args.validation_prompt.format(
        new_token.identifier,
    )
    # Dataset and DataLoaders creation:
    train_dataset = SdxlDataset(
        data_root=args.train_data_dir,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
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

    # Prepare everything with our `accelerator`.
    if (args.peft_rank > 0 or args.use_oft) and args.peft_start_step >= 0:
        # We'll prepare unet later when we switch to LoRA/OFT training
        text_encoder_1, text_encoder_2, optimizer, train_dataloader = (
            accelerator.prepare(
                text_encoder_1,
                text_encoder_2,
                optimizer,
                train_dataloader,
            )
        )
    else:
        # Standard preparation
        if args.peft_rank > 0 or args.use_oft:
            # Prepare unet with LoRA/OFT from the start
            (
                text_encoder_1,
                text_encoder_2,
                unet,
                optimizer,
                train_dataloader,
            ) = accelerator.prepare(
                text_encoder_1,
                text_encoder_2,
                unet,
                optimizer,
                train_dataloader,
            )
        else:
            (
                text_encoder_1,
                text_encoder_2,
                optimizer,
                train_dataloader,
            ) = accelerator.prepare(
                text_encoder_1,
                text_encoder_2,
                optimizer,
                train_dataloader,
            )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet and text_encoder_2 to device and cast to weight_dtype
    # Only move unet if it wasn't prepared (i.e., not doing LoRA from the start)
    # if not (args.peft_rank > 0 and args.peft_start_step == 0):
    #     unet.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

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
    peft_optimizer = None
    peft_start = False
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

    text_encoder_1.train()
    text_encoder_2.train()
    for step, batch in data_loop(train_dataloader, args.max_train_steps):
        # Convert images to latent space
        latents = (
            vae.encode(batch["pixel_values"].to(dtype=vae.dtype))
            .latent_dist.sample()
            .detach()
        )
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with accelerator.accumulate([text_encoder_1, text_encoder_2, unet]):
            # Get the text embedding for conditioning
            encoder_hidden_states_1 = (
                text_encoder_1(
                    batch["input_ids_1"],
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .to(dtype=weight_dtype)
            )
            encoder_output_2 = text_encoder_2(
                batch["input_ids_2"],
                output_attentions=False,
                output_hidden_states=True,
            )
            encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(
                dtype=weight_dtype
            )
            original_size = [
                (
                    batch["original_size"][0][i].item(),
                    batch["original_size"][1][i].item(),
                )
                for i in range(args.train_batch_size)
            ]
            crop_top_left = [
                (
                    batch["crop_top_left"][0][i].item(),
                    batch["crop_top_left"][1][i].item(),
                )
                for i in range(args.train_batch_size)
            ]
            target_size = (args.resolution, args.resolution)
            add_time_ids = torch.cat(
                [
                    torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                    for i in range(args.train_batch_size)
                ]
            ).to(accelerator.device, dtype=weight_dtype)
            added_cond_kwargs = {
                "text_embeds": encoder_output_2[0],
                "time_ids": add_time_ids,
            }
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
            )

            # Predict the noise residual.
            model_pred = unet(
                noisy_latents.to(dtype=weight_dtype),
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Get the target for loss depending on the prediction type.
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if args.dcoloss_beta > 0.0:
                with torch.no_grad():
                    cross_attention_kwargs = {"scale": 0.0}
                    refer_pred = unet(
                        noisy_latents.to(dtype=weight_dtype),
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                loss_model = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )
                loss_refer = F.mse_loss(
                    refer_pred.float(), target.float(), reduction="mean"
                )
                diff = loss_model - loss_refer
                inside_term = -1 * args.dcoloss_beta * diff
                loss = -1 * F.logsigmoid(inside_term)
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # L2 regularization between current and initial embeddings (CrossInit).
            if args.reg_lambda > 0.0:
                emb_1 = embeddings_1.added_tokens_embeddings.weight
                emb_2 = embeddings_2.added_tokens_embeddings.weight
                added_ids_1 = embeddings_1.to_added_token_id(added_token_ids)
                added_ids_2 = embeddings_2.to_added_token_id(added_token_ids_2)
                reg_loss = F.mse_loss(
                    emb_1[added_ids_1], init_embeds.to(emb_1.dtype)
                ) + F.mse_loss(emb_2[added_ids_2], init_embeds_2.to(emb_2.dtype))
                loss = loss + args.reg_lambda * reg_loss

            accelerator.backward(loss)
            # Use the appropriate optimizer based on training stage
            if peft_start and peft_optimizer is not None:
                # Step both optimizers when doing joint training
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                peft_optimizer.step()
                peft_optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            images = []
            global_step += 1

            # Switch to LoRA/OFT training if we've reached peft_start_step
            if (
                (args.peft_rank > 0 or args.use_oft)
                and args.peft_start_step > 0
                and global_step >= args.peft_start_step
                and not peft_start
            ):
                logger.info(
                    f"Switching to UNet {'OFT' if args.use_oft else 'LoRA'} fine-tuning at step {global_step}. "
                    f"Token embeddings will continue training alongside {'OFT' if args.use_oft else 'LoRA'}."
                )
                peft_start = True

                # Unfreeze and enable PEFT parameters
                for name, param in unet.named_parameters():
                    if args.use_oft:
                        if "oft" in name:
                            param.requires_grad_(True)
                    elif "lora" in name:
                        param.requires_grad_(True)

                # Create separate optimizer for PEFT parameters only
                peft_params = [
                    p
                    for n, p in unet.named_parameters()
                    if (
                        (args.use_oft and "oft" in n)
                        or (not args.use_oft and "lora" in n)
                    )
                    and p.requires_grad
                ]

                peft_optimizer = torch.optim.AdamW(
                    peft_params,
                    lr=args.peft_learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

                # Prepare PEFT optimizer and unet with accelerator
                unet, peft_optimizer = accelerator.prepare(unet, peft_optimizer)

                # Set UNet to training mode
                unet.train()
                logger.info(
                    f"PEFT training started with {len(peft_params)} parameters, "
                    "continuing embedding training with separate optimizer"
                )

            if global_step % args.save_steps == 0:
                weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                save_path = Path(args.output_dir) / weight_name
                save_embeddings(
                    accelerator.unwrap_model(text_encoder_1),
                    new_tokens,
                    save_path,
                    safe_serialization=True,
                )
                weight_name = f"learned_embeds_2-steps-{global_step}.safetensors"
                save_path = Path(args.output_dir) / weight_name
                save_embeddings(
                    accelerator.unwrap_model(text_encoder_2),
                    new_tokens_2,
                    save_path,
                    safe_serialization=True,
                )
                if (args.peft_rank > 0 or args.use_oft) and peft_start:
                    logger.info("Saving PEFT weights...")
                    save_sdxl_peft(
                        accelerator.unwrap_model(unet),
                        Path(args.output_dir)
                        / f"{'oft' if args.use_oft else 'lora'}-steps-{global_step}",
                    )

            if accelerator.is_main_process:
                if (
                    args.validation_prompt is not None
                    and global_step % args.validation_steps == 0
                ):
                    images = log_validation(
                        accelerator,
                        args,
                        text_encoder_1,
                        text_encoder_2,
                        tokenizer_1,
                        tokenizer_2,
                        unet,
                        vae,
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

            train_logger.log(
                global_step=global_step,
                loss=loss.detach().item(),
                embeddings_list=[embeddings_1, embeddings_2],
                token_ids_list=[added_token_ids, added_token_ids_2],
                lr=args.peft_learning_rate if peft_start else args.learning_rate,
                extra={"stage": "lora" if peft_start else "embedding"},
            )

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(
                accelerator,
                args,
                text_encoder_1,
                text_encoder_2,
                tokenizer_1,
                tokenizer_2,
                unet,
                vae,
                weight_dtype,
            )

        # Save the newly trained embeddings.
        weight_name = "learned_embeds.safetensors"
        save_path = Path(args.output_dir) / weight_name
        save_embeddings(
            accelerator.unwrap_model(text_encoder_1),
            new_tokens,
            save_path,
            safe_serialization=True,
        )
        weight_name = "learned_embeds_2.safetensors"
        save_path = Path(args.output_dir) / weight_name
        save_embeddings(
            accelerator.unwrap_model(text_encoder_2),
            new_tokens_2,
            save_path,
            safe_serialization=True,
        )

        # Save LoRA/OFT weights if LoRA/OFT training was performed
        if (args.peft_rank > 0 or args.use_oft) and peft_start:
            logger.info(f"Saving {'OFT' if args.use_oft else 'LoRA'} weights...")
            save_sdxl_peft(
                accelerator.unwrap_model(unet),
                Path(args.output_dir)
                / f"{'oft' if args.use_oft else 'lora'}-steps-{global_step}",
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
