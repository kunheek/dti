#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from PIL import Image
from scipy.spatial import distance
from torchvision.transforms import v2
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from dti.utils import load_embedding

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-e", "--exp_dir", type=str, required=True)

    parser.add_argument("--train_data", type=str, default="data/styledrop.json")
    parser.add_argument("--eval_prompt", type=str, default="data/style_prompt.txt")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--skip_gen", action="store_true")
    args = parser.parse_args()
    if args.exp_dir.endswith("/"):
        args.exp_dir = args.exp_dir[:-1]
    return args


def load_ti_embedding(embedding, embedding_2):
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL,
        subfolder="text_encoder",
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        MODEL,
        subfolder="tokenizer_2",
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        MODEL,
        subfolder="text_encoder_2",
    )

    identifier = load_embedding(
        tokenizer,
        text_encoder,
        embedding,
    )

    identifier = load_embedding(
        tokenizer_2,
        text_encoder_2,
        embedding_2,
    )

    return (
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        identifier,
    )


def is_live(subject):
    return subject in ("cat", "dog")


def generate_samples(
    pipeline,
    identifier,
    subject,
    eval_prompt,
    sample_dir,
    dtype=torch.float16,
):
    with open(eval_prompt, "r") as f:
        prompts = f.read().splitlines()
    print(prompts)

    shape = (1, 4, 128, 128)
    generator = torch.Generator("cuda")
    pipeline = pipeline.to(dtype=dtype)
    # seeds = (0, 1, 2, 3, 4, 5, 6, 7)
    seeds = (0, 1)
    latents = []
    for seed in seeds:
        generator.manual_seed(seed)
        z = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
        latents.append(z)
    latents = torch.cat(latents, dim=0)

    for prompt in prompts:
        filename = prompt.replace(" ", "_").format(subject)

        if isinstance(identifier, list):
            p = prompt.format(identifier[0])
        elif isinstance(identifier, str):
            p = prompt.format(identifier)
        images = pipeline(
            prompt=[p] * latents.shape[0],  # Duplicate prompt
            latents=latents,
        ).images
        for image, seed in zip(images, seeds):
            seed_dir = Path(sample_dir) / f"{seed}"
            seed_dir.mkdir(exist_ok=True)
            image.save(seed_dir / f"{filename}.png")


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, sample_dir, transform):
        # sample_dir/SEED/PROMPT.png
        self.samples = glob.glob(str(Path(sample_dir) / "*" / "*.png"))
        # self.samples = glob.glob(str(Path(sample_dir) / "*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = Path(sample).name.split(".")[0]
        prompt = f"{prompt.replace('_', ' ')}."
        image = Image.open(sample)
        if isinstance(self.transform, CLIPProcessor):
            image = self.transform(
                images=image,
                return_tensors="pt",
            )["pixel_values"][0]
        else:
            image = self.transform(image)
        return image, prompt


def evaluate(train_data, sample_dir):
    # Load DINO v2
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2.eval().requires_grad_(False).cuda()
    dino_transform = v2.Compose(
        [
            v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Load CLIP
    clip_model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    preprocess = CLIPProcessor.from_pretrained(clip_model_name)
    clip = CLIPModel.from_pretrained(clip_model_name).to(device="cuda")
    clip.eval().requires_grad_(False)

    scores = {"image": {}, "text": {}}
    for name in train_data:
        scores["image"][name] = []
        scores["text"][name] = []

        data_path = train_data[name]["path"]
        image_files = list(Path(data_path).iterdir())
        train_images = []
        for file in image_files:
            ext = file.split(".")[-1]
            if ext not in ("png", "jpg", "jpeg"):
                print(f"Invalid image file: {file}")
                continue
            image = Image.open(Path(data_path) / file)
            train_images.append(image)

        dino_images = [dino_transform(img) for img in train_images]
        dino_images = torch.stack(dino_images).cuda()
        dino_features = dinov2(dino_images).float().cpu().numpy()

        clip_images = preprocess(
            images=[img for img in train_images],
            return_tensors="pt",
        )["pixel_values"].to(device="cuda")
        clip_features = clip.get_image_features(clip_images)
        clip_features = F.normalize(clip_features, p=2, dim=-1)

        target_dir = Path(sample_dir) / name
        dataset = EvalDataset(target_dir, dino_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
        )

        dataset2 = EvalDataset(target_dir, preprocess)
        dataloader2 = torch.utils.data.DataLoader(
            dataset2,
            batch_size=16,
            shuffle=False,
        )

        for (image, _), (image2, prompt) in zip(dataloader, dataloader2):
            image = image.cuda()
            # Compute image-image similarity.
            features = dinov2(image).float().cpu().numpy()
            # image_similarity = 1.0 - torch.cdist(features, dino_features)  # B, N
            cos_dist = distance.cdist(features, dino_features, "cosine")
            image_similarity = 1.0 - cos_dist
            image_score = image_similarity.mean(axis=-1)
            scores["image"][name].append(image_score)

            # Compute image-text similarity.
            image2 = image2.cuda()
            features = clip.get_image_features(image2).float()
            text_inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(
                device="cuda"
            )
            text_features = clip.get_text_features(**text_inputs).float()
            text_score = F.cosine_similarity(features, text_features, dim=-1)
            scores["text"][name].append(text_score)

        scores["image"][name] = np.concatenate(scores["image"][name]).mean()
        scores["text"][name] = torch.cat(scores["text"][name]).mean().cpu().item()
        print(f"{name} image score: {scores['image'][name]:.4f}")
        print(f"{name} text score: {scores['text'][name]:.4f}")
    return scores


@torch.inference_mode()
def main():
    args = parse_arguments()

    exp_name = Path(args.exp_dir).name
    sample_dir = Path(args.out_dir) / exp_name
    if args.checkpoint is not None:
        sample_dir += f"-{args.checkpoint}"
    print(sample_dir)
    with open(args.train_data, "r") as f:
        data = json.load(f)

    if not args.skip_gen:
        vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
        for name in data:
            instance_dir = Path(args.exp_dir) / name
            print(name)
            # subject = data[name]["subject"]
            subject = data[name]["class"]
            live = subject in ("cat", "dog")
            eval_prompt = args.live_prompt if live else args.object_prompt

            embedding = Path(instance_dir) / "learned_embeds.safetensors"
            embedding_2 = Path(instance_dir) / "learned_embeds_2.safetensors"
            if args.checkpoint is not None:
                embedding = embedding.replace(
                    ".safetensors", f"-steps-{args.checkpoint}.safetensors"
                )
                embedding_2 = embedding_2.replace(
                    ".safetensors", f"-steps-{args.checkpoint}.safetensors"
                )
            (
                tokenizer,
                text_encoder,
                tokenizer_2,
                text_encoder_2,
                identifier,
            ) = load_ti_embedding(embedding, embedding_2)
            print("Identifier:", identifier)

            dtype = torch.float16
            pipeline = DiffusionPipeline.from_pretrained(
                MODEL,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2.to(dtype),
                unet=unet.to(dtype),
                vae=vae.to(dtype),
                weight_dtype=dtype,
            ).to("cuda", dtype=dtype)

            scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            pipeline.scheduler = scheduler

            file_list = list(Path(instance_dir).iterdir())
            print(file_list)
            if "pytorch_lora_weights.safetensors" in file_list:
                pipeline.load_lora_weights(instance_dir)
                print("Loaded LoRA weights.")
            else:
                print("No LoRA weights found.")

            generate_samples(
                pipeline,
                identifier,
                subject,
                eval_prompt,
                Path(sample_dir) / name,
                dtype=dtype,
            )

            pipeline.unload_lora_weights()

    scores = evaluate(data, sample_dir)
    print(scores)
    if args.checkpoint is not None:
        filename = f"score-{args.checkpoint}.json"
    else:
        filename = "score.json"
    with open(Path(args.exp_dir) / filename, "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
