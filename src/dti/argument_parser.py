"""Shared argument parsing."""

import argparse
import os

from dti.utils import str2bool


def add_base_args(parser):
    """Add common arguments across all training scripts."""
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=10,
        help="Log training metrics every N global steps.",
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
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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


def add_dti_args(parser):
    """Add DTI-specific arguments."""
    parser.add_argument(
        "--decompose_scale",
        type=str2bool,
        default=True,
        help="Whether to decompose the scale for the new token embeddings.",
    )
    parser.add_argument(
        "--token_scale",
        type=str,
        default="mean",
        help=(
            "The initial scale value for the new token embeddings when using the reparametrization trick. This is used"
            " to control the initial magnitude of the new token embeddings.",
        ),
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="token",
        choices=("token", "random", "mean", "xinit"),
        help=(
            "The method to use for initializing the new token embeddings. Choose between 'token', 'random', 'mean'"
            " and 'xinit'. 'token' uses the embedding of the initializer token, 'random' uses a random embedding,"
            " 'mean' uses the mean of all embeddings, and 'xinit' (CrossInit) uses the contextual representation"
            " of the initializer token from the text encoder's last hidden state."
        ),
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
        "--beta",
        type=float,
        default=0.0,
        help=(
            "The beta parameter for RSGD optimizer. This is used to control the momentum of the optimizer."
        ),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="RSGD Weight decay to use.",
    )
    parser.add_argument(
        "--train_magnitude",
        type=str2bool,
        default=False,
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
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.0,
        help=(
            "L2 regularization coefficient between current and initial embeddings. "
            "Used for CrossInit experiments. If 0.0 (default), no regularization is applied."
        ),
    )


def add_peft_args(parser):
    """Add PEFT-specific arguments."""
    parser.add_argument(
        "--peft_rank",
        type=int,
        default=0,
        help="If greater than 0, use LoRA/OFT with the specified rank.",
    )
    parser.add_argument(
        "--peft_modules",
        type=str,
        nargs="+",
        default=("to_q", "to_k", "to_v"),
        help="The modules to apply lora/oft to.",
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
        help="The step at which to start training the lora layers. If > 0, token embeddings will be trained first, then switched to UNet LoRA fine-tuning.",
    )
    parser.add_argument(
        "--use_oft",
        action="store_true",
        help="Whether to use OFT instead of LoRA.",
    )
    parser.add_argument(
        "--oft_r",
        type=int,
        default=4,
        help="r parameter for OFT.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="The gamma parameter for SNR weighting. If > 0.0, SNR weighting will be used during training.",
    )
    parser.add_argument(
        "--dcoloss_beta",
        type=float,
        default=0.0,  # 1000.0 to use DCO loss.
        help="The beta parameter for the DCO loss. If > 0.0, DCO loss will be used during training.",
    )


def add_optimizer_args(parser):
    """Add optimizer arguments."""
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rsgd",
        choices=["rsgd", "adamw"],
        help="The optimizer to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,  # 5e-3 for TI.
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )


def add_sana_specific_args(parser):
    """Add Sana-specific arguments."""
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--complex_human_instruction",
        type=str,
        default=None,
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )


def parse_sdxl_args() -> argparse.Namespace:
    """Parse SDXL-specific arguments."""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    add_base_args(parser)
    add_dti_args(parser)
    add_peft_args(parser)
    add_optimizer_args(parser)

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.num_vectors is None and args.initializer_token is None:
        raise ValueError(
            "You must specify either --num_vectors or --initializer_token."
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def parse_sana_args() -> argparse.Namespace:
    """Parse Sana-specific arguments."""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    add_base_args(parser)
    add_sana_specific_args(parser)
    add_dti_args(parser)
    add_optimizer_args(parser)

    # Override defaults for Sana
    parser.set_defaults(
        max_train_steps=1000,
        checkpointing_steps=100,
        kappa=5e-5,
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

    return args
