#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

from dti.utils import find_free_port


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OFT on SDXL")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-r", "--resolution", type=int, default=768)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--skip_training", action="store_true")
    # Basic training parameters.
    parser.add_argument("--total_steps", type=int, default=150)
    # OFT-specific parameters.
    parser.add_argument("--peft_lr", type=float, default=5e-5)
    parser.add_argument("--oft_r", type=int, default=4)  # following the paper.
    # Ablations.
    parser.add_argument("--init_method", type=str, default="token")

    args = parser.parse_args()

    args.model = "stabilityai/stable-diffusion-xl-base-1.0"
    args.expname = "oft-sdxl"
    args.runname = "oft-sdxl"
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

    val_step = args.total_steps // 5

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
        init_token = f"sks {cls}"

        cmd = [
            "scripts/train_sdxl.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--train_data_dir={data_path}",
            f"--output_dir=./{outdir}/{name}",
            "--learnable_property=object",
            f"--placeholder_token=<{name}>",
            f"--initializer_token={init_token}",
            "--mixed_precision=bf16",
            f"--resolution={args.resolution}",
            f"--save_steps={val_step}",
            f"--validation_steps={val_step}",
            "--validation_prompt=a {} with Japanese modern city street in the background",
            "--train_batch_size=4",
            "--gradient_accumulation_steps=1",
            "--seed=42",
            f"--init_method={args.init_method}",  # token, random, mean
            f"--max_train_steps={args.total_steps}",
            "--learning_rate=0.0",  # NOTE: we do not train the embedding.
            # Disable DTI-specific args.
            "--optimizer=adamw",
            "--decompose_scale=false",
            "--kappa=0.0",
            # Enable OFT
            "--use_oft",
            "--oft_r=4",
            "--peft_start_step=1",
            f"--peft_learning_rate={args.peft_lr}",
            "--dcoloss_beta=0.0",  # disable dcoloss
        ]

        # Save cmd as text file.
        os.makedirs(f"{outdir}/{name}", exist_ok=True)
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)

        subprocess.run(accelerate_cmd + cmd)

    # Evaluation.
    for ckpt in range(args.total_steps, val_step - 1, -val_step):
        cmd = [
            "python",
            "scripts/evaluate.py",
            f"-e={outdir}",
            f"--checkpoint={ckpt}",
        ]
        if args.instances is not None:
            cmd += ["--instances"] + args.instances
        subprocess.run(cmd)

    # Complex prompts (currently disabled).
    # cmd = [
    #     "python",
    #     "scripts/evaluate.py",
    #     f"-e={outdir}",
    #     "--prompt_set=complex",
    #     "--out_dir=images",
    # ]
    # subprocess.run(cmd)


if __name__ == "__main__":
    main()
