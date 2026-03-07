#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

from dti.constants import DIFFUSERS_MODELS
from dti.utils import find_free_port


def parse_arguments():
    parser = argparse.ArgumentParser(description="DTI on Wan (image-as-1-frame-video)")
    parser.add_argument("-g", "--gpu", type=str, default="2")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=("wan2.1_t2v_1.3b",),
        default="wan2.1_t2v_1.3b",
    )
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    # Basic training parameters.
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    # DTI-specific parameters.
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--scale", type=str, default="mean")
    parser.add_argument("--kappa", type=float, default=5e-5)
    parser.add_argument(
        "--init_method",
        type=str,
        default="token",
        choices=("token", "random", "mean", "xinit"),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square resolution used for image-as-video training.",
    )
    args = parser.parse_args()

    model = args.model.lower()
    args.model = DIFFUSERS_MODELS.get(model, None)
    if args.model is None:
        raise ValueError(f"Model {model} not found in DIFFUSERS_MODELS.")

    args.runname = f"dti-{model}"
    args.expname = f"dti-{model}"
    if args.init_method != "token":
        args.expname += f"-{args.init_method}"
    return args


def main():
    args = parse_arguments()

    with open("data/dreambooth.json", "r") as f:
        data = json.load(f)

    if args.instances is not None:
        for key in list(data):
            if key not in args.instances:
                del data[key]

    outdir = f"outputs/{args.expname}"
    if args.desc is not None:
        outdir += f"-{args.desc}"
    os.makedirs(outdir, exist_ok=True)

    val_freq = 100

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
        data_path = metadata["path"]
        init_token = metadata["initialization"]

        cmd = [
            "scripts/train_wan.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--train_data_dir={data_path}",
            f"--output_dir=./{outdir}/{name}",
            "--learnable_property=object",
            f"--placeholder_token=<{name}>",
            "--num_vectors=1",
            f"--initializer_token={init_token}",
            f"--resolution={args.resolution}",
            "--num_frames=1",
            f"--save_steps={val_freq}",
            f"--validation_steps={val_freq}",
            "--validation_prompt=a {} on a beach",
            f"--train_batch_size={args.batch_size}",
            f"--gradient_accumulation_steps={args.accum}",
            f"--max_train_steps={args.total_steps}",
            f"--learning_rate={args.lr}",
            "--scale_lr",
            f"--beta={args.beta}",
            f"--token_scale={args.scale}",
            f"--kappa={args.kappa}",
            f"--init_method={args.init_method}",
            "--seed=42",
            "--guidance_scale=5.0",
        ]

        os.makedirs(f"{outdir}/{name}", exist_ok=True)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write("\n".join(cmd))

        subprocess.run(accelerate_cmd + cmd)

    # Evaluate by generating and scoring first frames.
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
