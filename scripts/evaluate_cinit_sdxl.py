#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from safetensors.torch import load_file

from dti.constants import DIFFUSERS_MODELS
from dti.metrics import DINOv2Similarity, CLIPScore, SigLIPScore
from dti.metrics.dataset import MetricDataset, collate_eval_batch
from dti.metrics.prompts import ALL_PROMPT_SETS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-e", "--exp_dir", type=str, required=True)
    parser.add_argument("--no_ti", action="store_true")

    parser.add_argument("--train_data", type=str, default="data/dreambooth.json")
    parser.add_argument("--mask_dir", type=str, default="data/dreambooth_mask")
    parser.add_argument("--instances", type=str, nargs="+", default=None)

    parser.add_argument(
        "--prompt_set", type=str, default="simple", choices=["simple", "complex"]
    )
    # TODO: remove live/object prompts in the future.
    parser.add_argument("--live_prompt", type=str, default="data/live_prompts.txt")
    parser.add_argument("--object_prompt", type=str, default="data/object_prompts.txt")

    parser.add_argument("--style", action="store_true")
    parser.add_argument("--style_prompt", type=str, default="data/style_prompts.txt")

    parser.add_argument("--out_dir", type=str, default="images")
    parser.add_argument("--checkpoint", type=int, default=None)
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
        default="google/siglip2-large-patch16-512",
        help="Image-text similarity model to use (CLIP or SigLIP)",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Batch size for evaluation"
    )

    args = parser.parse_args()
    if args.exp_dir.endswith("/"):
        args.exp_dir = args.exp_dir[:-1]
    return args


def load_embedding(
    tokenizer, text_encoder, state_dict, placeholder=None, joiner: str = ""
) -> list[str]:
    assert Path(state_dict).exists(), f"File not found: {state_dict}"
    if not isinstance(state_dict, dict):
        state_dict = load_file(state_dict)

    # Load multiple token embeddings.
    identifiers = []
    for key, embs in state_dict.items():
        if placeholder is not None:
            key = placeholder
        tokens = [key]
        for i in range(1, embs.size(0)):
            tokens.append(f"{key}_v{i}")
        tokenizer.add_tokens(tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embeddings = text_encoder.get_input_embeddings().weight.data
        for id, emb in zip(token_ids, embs):
            embeddings[id] = emb.clone()
        identifier = joiner.join(tokens)
        identifiers.append(identifier)
    return identifiers


def load_embeddings(pipeline, embeddings):
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
        identifier = load_embedding(tokenizer, text_encoder, embed)
        identifiers.append(identifier)
        tokenizer = None
        text_encoder = None
    return identifiers


def is_live(superclass):
    return superclass in ("cat", "dog")


def generate_samples(
    pipeline,
    identifier,
    subject,
    prompts,
    sample_dir,
    device="cuda",
    dtype=torch.bfloat16,
):
    if isinstance(prompts, str) and Path(prompts).exists():
        with open(prompts, "r") as f:
            prompts = f.read().splitlines()
    elif isinstance(prompts, list):
        pass
    else:
        raise ValueError("Invalid prompts input.")
    print(prompts)

    pipeline = pipeline.to(dtype=dtype)
    # seeds = (0, 1, 2, 3, 4, 5, 6, 7)
    seeds = (0, 1, 2, 3)
    generators = []
    for seed in seeds:
        generators.append(torch.Generator(device).manual_seed(seed))

    for prompt in prompts:
        print(prompt)
        filename = prompt.replace(" ", "_").format(subject)

        if isinstance(identifier, list):
            p = prompt.format(identifier[0])
        elif isinstance(identifier, str):
            p = prompt.format(identifier)
        images = pipeline(
            prompt=[p] * len(generators),  # Duplicate prompt
            generator=generators,
        ).images
        for image, seed in zip(images, seeds):
            seed_dir = Path(sample_dir) / f"{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            image.save(seed_dir / f"{filename}.png")


def create_metrics(
    device=None,
    dinov2_model_size="small",
    image_text_model="google/siglip2-large-patch16-512",
):
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
    train_data,
    sample_dir,
    mask_dir=None,
    device=None,
    dinov2_model_size="small",
    image_text_model="google/siglip2-large-patch16-512",
    batch_size=8,
):
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

        # Load generated images with processors
        target_dir = Path(sample_dir) / name
        if not target_dir.exists():
            print(f"Warning: Sample directory not found for {name}: {target_dir}")
            scores["image"][name] = 0.0
            scores["text"][name] = 0.0
            continue

        # Create dataset with both processors
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
    sample_dir = Path(args.out_dir) / exp_name
    if args.checkpoint is not None:
        sample_dir = Path(str(sample_dir) + f"-{args.checkpoint}")
    print(sample_dir)
    if args.style:
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
            instance_dir = Path(args.exp_dir) / name / "learned_embeddings"
            print(f"instance_dir: {instance_dir}")
            subject = data[name]["class"]
            live = is_live(subject)
            if args.style:
                eval_prompt = args.style_prompt
            else:
                eval_prompt = (
                    prompt_sets.live_prompts if live else prompt_sets.object_prompts
                )
                # eval_prompt = args.live_prompt if live else args.object_prompt

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

            identifiers = load_embeddings(pipeline, embeddings)

            # If LoRA.
            file_list = list(instance_dir.iterdir())
            # print([f.name for f in file_list])
            if any(f.name == "pytorch_lora_weights.safetensors" for f in file_list):
                pipeline.load_lora_weights(str(instance_dir))
                print("Loaded LoRA weights.")
            else:
                print("No LoRA weights found.")

            generate_samples(
                pipeline,
                identifiers[0],
                subject,
                eval_prompt,
                str(sample_dir / name),
                device,
            )

            pipeline.unload_lora_weights()

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
    filename = "score"
    if args.checkpoint is not None:
        filename += f"-{args.checkpoint}"
    if args.mask_dir is not None:
        filename += "-masked"
    with open(Path(args.exp_dir) / f"{filename}.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
