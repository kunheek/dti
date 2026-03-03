#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

from dti.utils import find_free_port


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTI on SDXL")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-r", "--resolution", type=int, default=768)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--skip_training", action="store_true")
    # Basic training parameters.
    parser.add_argument("--total_steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    # DTI-specific parameters.
    parser.add_argument("--kappa", type=float, default=1e-4)
    parser.add_argument("--scale", type=str, default="mean")
    parser.add_argument("--beta", type=float, default=0.0)
    # Ablations.
    parser.add_argument("--decompose_scale", type=str, default="true")
    parser.add_argument("--init_method", type=str, default="token")
    parser.add_argument("--train_magnitude", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    args = parser.parse_args()

    args.model = "stabilityai/stable-diffusion-xl-base-1.0"
    args.expname = "dti-sdxl"
    return args


def main():
    args = parse_arguments()

    # full_data = "data/ti.json"
    full_data = "data/dreambooth.json"
    # full_data = "data/selected_data.json"
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
    val_step = 50

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
        init_token = cls  # NOTE: this is the default option.
        # init_token = metadata["initialization"]

        cmd = [
            # "scripts/train_sdxl_abal.py",
            "scripts/train_sdxl.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--train_data_dir={data_path}",
            # "--train_data_dir=data/dreambooth.json",
            # f"--instance={name}",
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
            f"--max_train_steps={args.total_steps}",
            f"--learning_rate={args.lr}",  # RSGD.
            "--scale_lr",
            "--seed=42",
            f"--init_method={args.init_method}",  # token, random, mean
            f"--token_scale={args.scale}",
            f"--kappa={args.kappa}",
            f"--decompose_scale={args.decompose_scale}",
            # "--zero_pad",  # TODO: check if this is necessary.
            f"--beta={args.beta}",  # RSGD beta.
            "--peft_rank=0",
        ]
        if args.adamw:
            cmd += [
                "--optimizer=adamw",
                # "--learning_rate=5e-4",  # AdamW.
            ]
        if args.train_magnitude:
            cmd += ["--train_magnitude"]

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

    # Complex prompts. (currently disabled).
    # cmd = [
    #     "python",
    #     "scripts/evaluate.py",
    #     f"-e={outdir}",
    #     "--prompt_set=complex_lv1",
    #     "--out_dir=images",
    # ]
    # subprocess.run(cmd)

    # cmd = [
    #     "python",
    #     "scripts/evaluate.py",
    #     f"-e={outdir}",
    #     "--prompt_set=complex_lv2",
    #     "--out_dir=images",
    # ]
    # subprocess.run(cmd)


if __name__ == "__main__":
    main()
