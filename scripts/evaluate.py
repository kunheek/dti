#!/usr/bin/env python3
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.training_utils import free_memory
from peft import PeftConfig, set_peft_model_state_dict, PeftModel
from PIL import Image
from safetensors.torch import load_file

from dti.constants import DIFFUSERS_MODELS
from dti.metrics import CLIPScore, DINOv2Similarity, SigLIPScore
from dti.metrics.dataset import MetricDataset, collate_eval_batch
from dti.metrics.prompts import ALL_PROMPT_SETS
from dti.utils import load_embedding


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-e", "--exp_dir", type=str, required=True)
    parser.add_argument("--no_ti", action="store_true")

    parser.add_argument("--train_data", type=str, default="data/dreambooth.json")
    parser.add_argument("--mask_dir", type=str, default="data/dreambooth_mask")
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="images")
    parser.add_argument("--checkpoint", type=int, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--prompt_set",
        type=str,
        default="simple",
        choices=ALL_PROMPT_SETS.keys(),
    )
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--skip_gen", action="store_true")

    # Metrics configuration
    parser.add_argument(
        "--dinov2_model",
        type=str,
        default="base",
        choices=["small", "base", "large", "giant"],
        help="DINOv2 model size for image similarity",
    )
    parser.add_argument(
        "--image_text_model",
        type=str,
        default="google/siglip-base-patch16-512",
        help="Image-text similarity model to use (CLIP or SigLIP)",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--rescale",
        type=str,
        default=None,
        choices=["min", "mean", "max"],
        help="Rescale loaded embeddings so their norm matches the given vocab statistic (min/mean/max). Default: no rescaling.",
    )

    args = parser.parse_args()
    if args.exp_dir.endswith("/"):
        args.exp_dir = args.exp_dir[:-1]

    # NOTE: remove after all experiments are done.
    if "sdxl" in args.exp_dir and args.gen_batch_size < 16:
        args.gen_batch_size = 16
    return args


def load_embeddings(
    pipeline: DiffusionPipeline,
    embeddings: list[Path],
    rescale: str | None = None,
) -> list[str]:
    identifiers = []
    for embed in embeddings:
        if "learned_embeds_2" in embed.stem:
            tokenizer = getattr(pipeline, "tokenizer_2")
            text_encoder = getattr(pipeline, "text_encoder_2")
        elif "learned_embeds_3" in embed.stem:
            tokenizer = getattr(pipeline, "tokenizer_3")
            text_encoder = getattr(pipeline, "text_encoder_3")
        elif "learned_embeds" in embed.stem:
            tokenizer = getattr(pipeline, "tokenizer")
            text_encoder = getattr(pipeline, "text_encoder")
        identifier = load_embedding(tokenizer, text_encoder, embed, rescale=rescale)
        identifiers.append(identifier)
        tokenizer = None
        text_encoder = None
    return identifiers


def is_live(superclass: str) -> bool:
    return superclass in ("cat", "dog")


def generate_samples(
    pipeline: DiffusionPipeline,
    identifier: str | list[str],
    subject: str,
    prompts: str | list[str] | object,
    sample_dir: str,
    seeds: list[int],
    batch_size: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    if isinstance(prompts, str) and Path(prompts).exists():
        with open(prompts, "r") as f:
            prompts = f.read().splitlines()
    elif isinstance(prompts, list):
        pass
    elif hasattr(prompts, "__iter__") and hasattr(prompts, "__getitem__"):
        # Handle StylePrompts or other iterable objects
        prompts = list(prompts)
    else:
        raise ValueError("Invalid prompts input.")
    # print(prompts)

    pipeline = pipeline.to(dtype=dtype)

    prompts_pipe = []
    seeds_pipe = []
    for prompt in prompts:
        for seed in seeds:
            prompts_pipe.append(prompt)
            seeds_pipe.append(seed)
    if isinstance(identifier, list):
        identifier = identifier[0]
    call_sig = inspect.signature(pipeline.__call__).parameters
    is_video_pipeline = "num_frames" in call_sig

    def to_pil_first_frame(video):
        frame = video[0] if isinstance(video, (list, tuple)) else video
        if isinstance(frame, Image.Image):
            return frame
        if torch.is_tensor(frame):
            arr = frame.detach().cpu().numpy()
        else:
            arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    for i in range(0, len(prompts_pipe), batch_size):
        # Sample pairs with given batch size.
        seeds_batch = seeds_pipe[i : i + batch_size]
        prompts_batch = prompts_pipe[i : i + batch_size]
        generators = [torch.Generator(device).manual_seed(seed) for seed in seeds_batch]

        call_kwargs = {
            "prompt": [p.format(identifier) for p in prompts_batch],
            "generator": generators,
        }
        if is_video_pipeline:
            call_kwargs["num_frames"] = 1
            call_kwargs["output_type"] = "pil"

        outputs = pipeline(**call_kwargs)
        if hasattr(outputs, "images"):
            images = outputs.images
        elif hasattr(outputs, "frames"):
            images = [to_pil_first_frame(video) for video in outputs.frames]
        else:
            raise ValueError(
                f"Unsupported pipeline output type: {type(outputs)}. "
                "Expected `images` or `frames`."
            )

        for image, prompt, seed in zip(images, prompts_batch, seeds_batch):
            filename = prompt.replace(" ", "_").format(subject)
            seed_dir = Path(sample_dir) / f"{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            image.save(seed_dir / f"{filename}.png")
    free_memory()


def create_metrics(
    device: str | torch.device | None = None,
    dinov2_model_size: str = "small",
    image_text_model: str = "google/siglip2-large-patch16-512",
) -> tuple[DINOv2Similarity, CLIPScore | SigLIPScore]:
    """Create and initialize metrics with error handling."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image_metric = DINOv2Similarity(model_size=dinov2_model_size, device=device)
        print(f"✓ DINOv2 metric initialized with model size: {dinov2_model_size}")
    except Exception as e:
        print(f"✗ Failed to initialize DINOv2 metric: {e}")
        raise

    try:
        if "clip" in image_text_model.lower():
            text_metric = CLIPScore(model_name=image_text_model, device=device)
        else:
            text_metric = SigLIPScore(model_name=image_text_model, device=device)
    except Exception as e:
        print(f"✗ Failed to initialize CLIP metric: {e}")
        raise

    return image_metric, text_metric


def evaluate(
    train_data: dict,
    sample_dir: Path,
    mask_dir: Path | None = None,
    device: str | torch.device | None = None,
    dinov2_model_size: str = "small",
    image_text_model: str = "google/siglip2-large-patch16-512",
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Evaluate generated images using the new metrics system."""

    # Initialize metrics
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        image_metric, text_metric = create_metrics(
            device,
            dinov2_model_size,
            image_text_model=image_text_model,
        )
    except Exception as e:
        print(f"Failed to initialize metrics: {e}")
        return {"image": {}, "text": {}}

    scores = {"image": {}, "text": {}}

    for name in train_data:
        print(f"Evaluating {name}...")

        # Load training/reference images
        data_path = train_data[name]["path"]
        image_files = list(Path(data_path).iterdir())
        train_images = []

        for file_path in image_files:
            if file_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                print(f"Invalid image file: {file_path.name}")
                continue

            image = Image.open(file_path).convert("RGB")  # Ensure RGB format
            if mask_dir is not None:
                mask_file = file_path.stem + ".png"
                mask_path = Path(mask_dir) / name / mask_file
                if mask_path.exists():
                    mask = Image.open(mask_path).convert("RGB")
                    image_np = np.asarray(image)
                    mask_np = np.asarray(mask) / 255.0
                    image_np = image_np * mask_np
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                else:
                    print(f"Mask not found: {mask_path}")
            train_images.append(image)

        if not train_images:
            print(f"Warning: No training images found for {name}")
            scores["image"][name] = 0.0
            scores["text"][name] = 0.0
            continue

        # Load generated images with processors.
        target_dir = Path(sample_dir) / name
        if not target_dir.exists():
            print(f"Warning: Sample directory not found for {name}: {target_dir}")
            scores["image"][name] = 0.0
            scores["text"][name] = 0.0
            continue

        # Create dataset with both processors.
        dataset = MetricDataset(str(target_dir))

        if len(dataset) == 0:
            print(f"Warning: No generated images found for {name}")
            scores["image"][name] = 0.0
            scores["text"][name] = 0.0
            continue

        # Create DataLoader with custom collate function
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_eval_batch,
        )

        # Store scores for this instance
        image_scores = []
        text_scores = []

        # Process in batches using DataLoader
        print(f"Processing {len(dataset)} generated images...")
        for batch_idx, batch_data in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}", end="\r", flush=True)

            # Compute image-image similarity using DINOv2
            # For each generated image, compare with all training images.
            similarities = image_metric(
                batch_data["images"], train_images, aggregate="row_mean"
            )
            mean_similarity = similarities.mean()
            image_scores.append(mean_similarity.item())

            # Compute image-text similarity using CLIP.
            similarities = text_metric(
                batch_data["images"], batch_data["texts"], aggregate="diag"
            )
            text_scores.extend(similarities.cpu().tolist())

        # Print newline after batch processing
        print()

        # Compute final scores for this instance
        scores["image"][name] = np.mean(image_scores) if image_scores else 0.0
        scores["text"][name] = np.mean(text_scores) if text_scores else 0.0

        print(f"{name} image score: {scores['image'][name]:.4f}")
        print(f"{name} text score: {scores['text'][name]:.4f}")

    return scores


@torch.inference_mode()
def main():
    args = parse_arguments()

    # set gpu
    device = f"cuda:{args.gpu}"
    print("================================")
    print(f"Using GPU: {args.gpu}")
    print(f"Device: {device}")
    print("================================")

    exp_name = Path(args.exp_dir).name
    sample_dir = Path(args.out_dir) / args.prompt_set / exp_name
    if args.checkpoint is not None:
        sample_dir = Path(str(sample_dir) + f"-{args.checkpoint}")
    if args.rescale is not None:
        sample_dir = Path(str(sample_dir) + f"-rescale_{args.rescale}")
    print(sample_dir)
    if args.prompt_set == "style":
        args.train_data = "data/styledrop.json"
    with open(args.train_data, "r") as f:
        full_data = json.load(f)
    if args.instances is not None:
        data = {key: full_data[key] for key in args.instances}
    else:
        data = full_data

    if not args.skip_gen:
        # exp_dir pattern: .../[method]-[model_key]-[desc1]-[desc2]...
        method, model_key, *desc = exp_name.split("-")
        # For legacy support.  TODO: remove in the future.
        if model_key == "sdxlbase":
            model_key = "sdxl"
        hf_model_name = DIFFUSERS_MODELS[model_key]
        pipeline = DiffusionPipeline.from_pretrained(hf_model_name)
        pipeline.to(device)

        prompt_sets = ALL_PROMPT_SETS[args.prompt_set]

        for name in data:
            instance_dir = Path(args.exp_dir) / name
            print(name)
            subject = data[name]["class"]
            live = is_live(subject)
            if args.prompt_set == "style":
                eval_prompt = prompt_sets
            else:
                eval_prompt = (
                    prompt_sets.live_prompts if live else prompt_sets.object_prompts
                )

            embeddings = []
            if args.checkpoint is not None:
                step_str = f"-steps-{args.checkpoint}"
            else:
                step_str = ""
            # Find all .safetensors, .bins in the instance directory.
            # Patterns: learned_embeds{step_str}.safetensors, learned_embeds{step_str}.bin
            for file in instance_dir.iterdir():
                if file.suffix in (".safetensors", ".bin") and file.stem in (
                    f"learned_embeds{step_str}",
                    f"learned_embeds_2{step_str}",
                    f"learned_embeds_3{step_str}",  # For SD 3.5
                ):
                    embeddings.append(file)
            embeddings = sorted(embeddings)
            if len(embeddings) == 0:
                print(
                    f"No embedding file found for {name} at checkpoint {args.checkpoint}. Skipping generation."
                )
                continue

            identifiers = load_embeddings(pipeline, embeddings, rescale=args.rescale)

            # If LoRA - load checkpoint-specific or final weights.
            peft_dir = None
            if args.checkpoint is not None:
                # Look for checkpoint-specific LoRA/OFT directory
                for prefix in ["lora", "oft"]:
                    peft_ckpt_dir = instance_dir / f"{prefix}-steps-{args.checkpoint}"
                    if peft_ckpt_dir.exists():
                        peft_dir = peft_ckpt_dir
                        print(f"Found PEFT checkpoint: {peft_ckpt_dir.name}")
                        break

            # Fall back to root directory for final LoRA weights
            if peft_dir is None:
                if (instance_dir / "pytorch_lora_weights.safetensors").exists():
                    peft_dir = instance_dir
                    print("Found final LoRA weights in root directory.")
                elif (instance_dir / "adapter_model.safetensors").exists():
                    peft_dir = instance_dir
                    print("Found final PEFT weights in root directory.")

            # Load LoRA weights if found
            if peft_dir is not None:
                try:
                    pipeline.load_lora_weights(str(peft_dir))
                    print(f"Loaded LoRA weights from: {peft_dir}")
                except Exception as e:
                    print(f"Failed to load LoRA weights with diffusers: {e}")
                    print("Attempting to load as PEFT model (OFT/LoRA)...")

                    # Get the denoiser model (unet for SDXL, transformer for Flux/Sana)
                    denoiser = getattr(pipeline, "unet", None) or getattr(
                        pipeline, "transformer", None
                    )

                    # Load the config and add the adapter first
                    peft_config = PeftConfig.from_pretrained(str(peft_dir))
                    denoiser.add_adapter(peft_config)

                    # Load state dict and set it
                    adapter_path = Path(peft_dir) / "adapter_model.safetensors"
                    state_dict = load_file(str(adapter_path))
                    set_peft_model_state_dict(denoiser, state_dict)
                    print(f"Loaded PEFT weights from: {peft_dir}")
            else:
                print("No PEFT weights found.")

            generate_samples(
                pipeline,
                identifiers[0],
                subject,
                eval_prompt,
                str(sample_dir / name),
                seeds=args.seeds,
                batch_size=args.gen_batch_size,
                device=device,
            )

            # Get the denoiser model (unet for SDXL, transformer for SD3/Flux/Sana)
            denoiser = getattr(pipeline, "unet", None) or getattr(
                pipeline, "transformer", None
            )

            if denoiser is not None and isinstance(denoiser, PeftModel):
                if hasattr(pipeline, "unet"):
                    pipeline.unet = pipeline.unet.unload()
                else:
                    pipeline.transformer = pipeline.transformer.unload()
            elif denoiser is not None and hasattr(denoiser, "peft_config"):
                # Unload adapter added via add_adapter
                denoiser.delete_adapters("default")
            else:
                pipeline.unload_lora_weights()

    print("===============================")
    print("sample_dir:", sample_dir)
    print("===============================")
    scores = evaluate(
        data,
        sample_dir,
        args.mask_dir,
        device,
        dinov2_model_size=args.dinov2_model,
        image_text_model=args.image_text_model,
        batch_size=args.eval_batch_size,
    )
    print(scores)
    filename = f"score_{args.prompt_set}"
    if args.checkpoint is not None:
        filename += f"-{args.checkpoint}"
    if args.mask_dir is not None:
        filename += "-masked"
    if args.rescale is not None:
        filename += f"-rescale_{args.rescale}"
    with open(Path(args.exp_dir) / f"{filename}.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
