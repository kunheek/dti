#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DiffusionPipeline

from dti.utils import load_embedding
from dti.clip_text_encoder import TextModel, TextModelWithProjection

SD = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run textual inversion experiment")
    parser.add_argument("-g", "--gpu", type=str, default="7")
    parser.add_argument("-e", "--exp", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--dti", action="store_true")
    args = parser.parse_args()
    return args


@torch.inference_mode()
def main():
    args = parse_arguments()

    if args.dti:
        text_encoder = TextModel.from_pretrained(SD, subfolder="text_encoder")
        mean_norm = text_encoder.get_mean_scale()

        text_encoder_2 = TextModelWithProjection.from_pretrained(
            SD, subfolder="text_encoder_2"
        )
        mean_norm_2 = text_encoder_2.get_mean_scale()
        pipe = DiffusionPipeline.from_pretrained(
            SD,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(SD)
    pipe.to(f"cuda:{args.gpu}", torch.float16)

    ckpt = Path(args.exp) / "learned_embeds.safetensors"
    ckpt_2 = Path(args.exp) / "learned_embeds_2.safetensors"
    identifier = load_embedding(pipe.tokenizer, pipe.text_encoder, ckpt)
    identifier_2 = load_embedding(pipe.tokenizer_2, pipe.text_encoder_2, ckpt_2)
    assert identifier == identifier_2
    if args.dti:
        text_encoder.get_mean_scale().weight.data[-1].copy_(mean_norm)
        text_encoder_2.get_mean_scale().weight.data[-1].copy_(mean_norm_2)
    toks = identifier.split()
    print(toks)
    prompts = [f"photo of a {identifier} in the snow."]
    for tok in toks:
        prompts.append(f"photo of a {tok} in the snow.")
    print(prompts)

    all_images = []
    for prompt in prompts:
        generator = torch.Generator().manual_seed(args.seed)
        images = pipe(
            prompt=prompt,
            generator=generator,
        ).images
        image = np.asarray(images[0])
        all_images.append(image)
    plt.imshow(np.concatenate(all_images, axis=1))
    plt.axis("off")
    plt.savefig("output.png")


if __name__ == "__main__":
    main()
