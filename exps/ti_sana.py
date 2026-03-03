#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

from dti.constants import DIFFUSERS_MODELS
from dti.utils import find_free_port


def parse_arguments():
    parser = argparse.ArgumentParser(description="TI on SANA")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=("sana1.5_1.6b", "sana1.5_4.8b"),
        default="sana1.5_1.6b",
    )
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--skip_training", action="store_true")
    # Basic training parameters.
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()

    model = args.model.lower()
    args.model = DIFFUSERS_MODELS.get(model, None)
    if args.model is None:
        raise ValueError(f"Model {model} not found in DIFFUSERS_MODELS.")
    args.resolution = 512 if "512" in args.model else 1024

    args.expname = f"ti-{model}"
    return args


def main():
    args = parse_arguments()

    full_data = "data/dreambooth.json"
    with open(full_data, "r") as f:
        full_data = json.load(f)

    data = {}
    if args.instances is not None:
        for key in args.instances:
            data[key] = full_data[key]
    else:
        data = full_data

    outdir = f"outputs/{args.expname}"
    if args.desc is not None:
        outdir += f"-{args.desc}"
    os.makedirs(outdir, exist_ok=True)

    val_freq = 100

    batch_size = args.batch_size // args.accum

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    accelerate_cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        f"--main_process_port={find_free_port()}",
    ]

    for name, metadata in data.items():
        if args.skip_training:
            print(f"Skipping training for {name}")
            break

        data_path = metadata["path"]
        cls = metadata["class"]
        init_token = cls  # NOTE: use class name as initializer token.

        cmd = [
            "scripts/train_sana.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--train_data_dir={data_path}",
            f"--output_dir=./{outdir}/{name}",
            "--learnable_property=object",
            f"--placeholder_token=<{name}>",
            f"--initializer_token={init_token}",
            f"--resolution={args.resolution}",
            f"--save_steps={val_freq}",
            f"--validation_steps={val_freq}",
            "--validation_prompt=a {} on a beach",
            f"--train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={args.accum}",
            f"--max_train_steps={args.total_steps}",
            f"--learning_rate={args.lr}",
            "--scale_lr",
            "--seed=42",
            # Disable DTI-specific args.
            "--optimizer=adamw",
            "--decompose_scale=false",
            "--kappa=0.0",
        ]

        # Save cmd as text file.
        os.makedirs(f"{outdir}/{name}", exist_ok=True)
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)

        subprocess.run(accelerate_cmd + cmd)

    # Evaluation.
    for ckpt in range(args.total_steps, val_freq - 1, -val_freq):
        cmd = [
            "python",
            "scripts/evaluate.py",
            f"-e={outdir}",
            f"--checkpoint={ckpt}",
        ]
        if args.instances is not None:
            cmd += ["--instances"] + args.instances
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
