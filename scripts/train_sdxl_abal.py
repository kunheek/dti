#!/usr/bin/env python3
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from tqdm.auto import tqdm

import transformers
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from diffusers.utils import is_wandb_available

from dti.datasets import SdxlDataset
from dti.training.manifold import (
    project_grads_to_tangent_space,
    retract_token_embeddings,
)
from dti.training.token_embedding_ops import (
    add_new_token,
    replace_token_embedding,
)
from dti.training_utils import (
    register_embedding_only_checkpoint_hooks,
    save_embeddings,
)
from dti.utils import str2bool

if is_wandb_available():
    import wandb
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="The name of the instance to use. If not specified, all instances in the train data directory will be used.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=None,
        help="Number of vectors to learn. The model will learn a vector for each placeholder token.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="How many times to repeat the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.015,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_eta_min",
        type=float,
        default=10.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optionally set the run name to use for logging.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--decompose_scale",
        type=str2bool,
        default=True,
        help=(
            "Whether to use reparametrization trick for the new token embeddings. This is used to improve the"
            " stability of the training process."
        ),
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="token",
        choices=["token", "random", "mean"],
        help=(
            "The method to use for initializing the new token embeddings. Choose between 'token', 'random' and 'mean'."
            " 'token' uses the embedding of the initializer token, 'random' uses a random embedding and 'mean' uses"
            " the mean of all embeddings."
        ),
    )
    parser.add_argument(
        "--init_scale",
        type=str,
        default="mean",
        help="",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1e-4,
        help=(
            "The concentration parameter for the von Mises-Fisher distribution. This is used to initialize the"
            " embeddings of the new token."
        ),
    )
    parser.add_argument(
        "--kappa_min",
        type=float,
        default=None,
        help=(
            "The minimum concentration parameter for the von Mises-Fisher distribution. This is used to initialize the"
            " embeddings of the new token."
        ),
    )
    parser.add_argument(
        "--train_magnitude",
        action="store_true",
        help=(
            "Whether to train the magnitude of the new token embeddings. If set, the magnitude of the new token"
            " embeddings will be trained."
        ),
    )
    parser.add_argument(
        "--zero_pad",
        action="store_true",
        help="Whether to pad the text with <pad> tokens.",
    )

    # LoRA.
    parser.add_argument(
        "--peft_rank",
        type=int,
        default=0,
        help="If greater than 0, use LoRA with the specified rank.",
    )
    parser.add_argument(
        "--peft_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v"],
        help="The modules to apply lora to.",
    )
    parser.add_argument(
        "--peft_learning_rate",
        type=float,
        default=1e-4,
        help="The learning rate for the lora layers.",
    )
    parser.add_argument(
        "--peft_start_step",
        type=int,
        default=0,
        help="The step at which to start training the lora layers.",
    )

    # Optimizer.
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rsgd",
        choices=["rsgd", "adamw"],
        help="The optimizer to use.",
    )

    parser.add_argument(
        "--use_adam",
        action="store_true",
        help="If true, use AdamW instead of RSGD.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.num_vectors is None and args.initializer_token is None:
        raise ValueError(
            "You must specify either --num_vectors or --initializer_token."
        )

    if args.kappa_min is None:
        args.kappa_min = args.kappa

    # TODO: remove in future.
    if args.use_adam:
        args.optimizer = "adamw"

    return args


def log_validation(
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    epoch,
    is_final_validation=False,
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
        unet=unet,
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
    print(args.validation_prompt)
    for _ in range(args.num_validation_images):
        image = pipeline(
            args.validation_prompt,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()
    return images


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir) / args.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizers.
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    # Replace text embeddings with DtiTokenEmbeddings.
    text_encoder_1 = replace_token_embedding(
        text_encoder_1,
        use_scale=args.decompose_scale,
    )
    text_encoder_2 = replace_token_embedding(
        text_encoder_2,
        use_scale=args.decompose_scale,
    )

    # Estimate kappa.
    # mu_1, kappa_1 = estimate_kappa(
    #     target=args.initializer_token,
    #     tokenizer=tokenizer_1,
    #     text_encoder=text_encoder_1,
    #     n=500,
    #     as_tensor=True,
    # )
    # mu_2, kappa_2 = estimate_kappa(
    #     target=args.initializer_token,
    #     tokenizer=tokenizer_2,
    #     text_encoder=text_encoder_2,
    #     n=500,
    #     as_tensor=True,
    # )
    # mu_1 = mu_1.to(accelerator.device)
    # mu_2 = mu_2.to(accelerator.device)
    # kappa_1 = kappa_1.to(accelerator.device)
    # kappa_2 = kappa_2.to(accelerator.device)
    # print(kappa_1, kappa_2)

    added_token_ids = []
    added_token_ids_2 = []
    new_tokens = []
    new_tokens_2 = []

    new_token, init_embeds = add_new_token(
        tokenizer_1,
        text_encoder_1,
        args.placeholder_token,
        num_vectors=args.num_vectors,
        init_scale=args.init_scale,
        init_token=args.initializer_token,
        init_method=args.init_method,
    )
    new_token_2, init_embeds_2 = add_new_token(
        tokenizer_2,
        text_encoder_2,
        args.placeholder_token,
        num_vectors=args.num_vectors,
        init_scale=args.init_scale,
        # init_token=mu_2,
        init_token=args.initializer_token,
        init_method=args.init_method,
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
    text_encoder_1.get_input_embeddings().requires_grad_(True)
    text_encoder_2.get_input_embeddings().requires_grad_(True)
    scale_grad = args.train_magnitude
    text_encoder_1.get_input_embeddings().scales.requires_grad_(scale_grad)
    text_encoder_2.get_input_embeddings().scales.requires_grad_(scale_grad)

    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()

    if args.peft_rank > 0:
        lora_config = LoraConfig(
            r=args.peft_rank,
            lora_alpha=args.peft_rank,
            init_lora_weights="gaussian",
            target_modules=args.peft_modules,
        )
        unet.add_adapter(lora_config)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        args.lr_eta_min = (
            args.lr_eta_min
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer.
    params_to_train = [
        text_encoder_1.get_input_embeddings().weight,
        text_encoder_2.get_input_embeddings().weight,
    ]
    if args.train_magnitude:
        scales = [
            text_encoder_1.get_input_embeddings().scales.weight,
            text_encoder_2.get_input_embeddings().scales.weight,
        ]
        params_to_train = [
            {"params": params_to_train},
            {"params": scales, "lr": args.learning_rate},
        ]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_train,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.SGD(
            params_to_train,
            lr=args.learning_rate,
            weight_decay=0.0,
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

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_train_steps,
            eta_min=args.lr_eta_min,  # minimum learning rate. Default is 10.
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
        )

    text_encoder_1.train()
    text_encoder_2.train()
    # Prepare everything with our `accelerator`.
    text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler
        )
    )
    register_embedding_only_checkpoint_hooks(
        accelerator,
        embedding_getters=[
            lambda model: model.get_input_embeddings(),
            lambda model: model.get_input_embeddings(),
        ],
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet and text_encoder_2 to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    # Afterwards we calculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(Path(args.output_dir) / path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        target_embeds = init_embeds / torch.linalg.norm(
            init_embeds, dim=-1, keepdim=True
        )
        target_embeds_2 = init_embeds_2 / torch.linalg.norm(
            init_embeds_2, dim=-1, keepdim=True
        )
    v1_t, v2_t = 0.0, 0.0

    # Keep original embeddings as reference.
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder_1)
        .get_input_embeddings()
        .weight.data.clone()
    )
    orig_embeds_params_2 = (
        accelerator.unwrap_model(text_encoder_2)
        .get_input_embeddings()
        .weight.data.clone()
    )
    index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
    index_no_updates[min(added_token_ids) : max(added_token_ids) + 1] = False
    index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool)
    index_no_updates_2[min(added_token_ids_2) : max(added_token_ids_2) + 1] = False

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()
        for step, batch in enumerate(train_dataloader):
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

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.optimizer == "rsgd":
                        with torch.no_grad():
                            # b2 = 1 - (global_step+1)**(-0.1)
                            v1_t = project_grads_to_tangent_space(
                                accelerator.unwrap_model(text_encoder_1),
                                added_token_ids,
                                kappa=args.kappa,
                                target_embeds=target_embeds,
                                t=global_step,
                                v_tm1=v1_t,
                                beta2=args.adam_beta2,
                            )
                            v2_t = project_grads_to_tangent_space(
                                accelerator.unwrap_model(text_encoder_2),
                                added_token_ids_2,
                                kappa=args.kappa,
                                target_embeds=target_embeds_2,
                                t=global_step,
                                v_tm1=v2_t,
                                beta2=args.adam_beta2,
                            )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                if accelerator.sync_gradients:
                    with torch.no_grad():
                        embeddings = accelerator.unwrap_model(
                            text_encoder_1
                        ).get_input_embeddings()
                        embeddings.weight[index_no_updates] = orig_embeds_params[
                            index_no_updates
                        ]
                        index_updates = ~index_no_updates
                        retract_token_embeddings(
                            embeddings,
                            index_updates=index_updates,
                        )

                        embeddings_2 = accelerator.unwrap_model(
                            text_encoder_2
                        ).get_input_embeddings()
                        embeddings_2.weight[index_no_updates_2] = orig_embeds_params_2[
                            index_no_updates_2
                        ]
                        index_updates_2 = ~index_no_updates_2
                        retract_token_embeddings(
                            embeddings_2,
                            index_updates=index_updates_2,
                        )

                        # Reinitialize the image tokens.
                        # max_id = max(placeholder_token_ids) + 1
                        # embeddings.weight[
                        #     max_id:
                        # ] = torch.randn_like(orig_embeds_params[max_id:]).mul(0.02)
                        # embeddings_2.weight[
                        #     max_id:
                        # ] = torch.randn_like(orig_embeds_params_2[max_id:]).mul(0.02)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
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
                            text_encoder_1,
                            text_encoder_2,
                            tokenizer_1,
                            tokenizer_2,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            epoch,
                        )
                        rows = args.num_validation_images // 2
                        cols = 2
                        image_grid = diffusers.utils.make_image_grid(
                            images, rows=rows, cols=cols
                        )
                        image_grid.save(
                            Path(args.output_dir) / f"validation-{global_step:04d}.jpg"
                        )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                # "cos_1": cos_sim.detach().item(),
                # "cos_2": cos_sim_2.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(
                text_encoder_1,
                text_encoder_2,
                tokenizer_1,
                tokenizer_2,
                unet,
                vae,
                args,
                accelerator,
                weight_dtype,
                epoch,
                is_final_validation=True,
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

    accelerator.end_training()


if __name__ == "__main__":
    main()
