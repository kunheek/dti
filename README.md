# Directional Textual Inversion (DTI)

<p align="center">
  <a href="https://kunheek.github.io/dti"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.13672"><img src="https://img.shields.io/badge/arXiv-2512.13672-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/kunheek/dti/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <strong>Directional Textual Inversion for Personalized Text-to-Image Generation</strong>
</p>

<p align="center">
  <a href="https://kunheek.github.io">Kunhee Kim</a><sup>1*</sup> ·
  <a href="https://nahyeonkaty.github.io">NaHyeon Park</a><sup>1*</sup> ·
  <a href="https://sites.google.com/sookmyung.ac.kr/aiv-lab-smwu/home">Kibeom Hong</a><sup>2</sup> ·
  <a href="https://kaist-cvml.github.io">Hyunjung Shim</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>KAIST · <sup>2</sup>Sookmyung Women's University
</p>

---

## Overview

Textual Inversion (TI) is efficient but often suffers from **embedding norm inflation**—learned tokens drift to out-of-distribution magnitudes, degrading prompt fidelity in pre-norm Transformers.

**Directional Textual Inversion (DTI)** addresses this by:

- **Decoupling magnitude and direction**: Fixing embedding magnitude to an in-distribution scale
- **Spherical optimization**: Optimizing only the direction on the unit hypersphere via Riemannian SGD
- **vMF prior**: Using a von Mises-Fisher prior for semantic coherence

DTI achieves superior text fidelity and subject preservation on **Stable Diffusion XL (SDXL)**, **SANA**, and **Wan2.1-T2V-1.3B** (image-as-1-frame-video setup).

Our implementation is built on top of the HuggingFace [`diffusers`](https://github.com/huggingface/diffusers) library and is fully compatible with existing Textual Inversion (TI) pipelines.

<p align="center">
  <img src="docs/assets/teaser.png" alt="DTI Results" width="100%">
</p>

## Installation

**Requirements:** Python 3.9+ · PyTorch 2.0+ · CUDA GPU

```bash
git clone https://github.com/kunheek/dti
cd dti
pip install -e .
```

<details>
<summary>Using [`uv`](https://docs.astral.sh/uv/) (recommended for faster setup)</summary>

```bash
uv venv python=3.12
source .venv/bin/activate
pip install -e .
```
</details>

## Quick Start

### Train DTI on SDXL

```bash
python exps/ours_sdxl.py -g 0
```

This runs DTI on all DreamBooth subjects. To train on a specific subject:

```bash
python exps/ours_sdxl.py -g 0 --instances dog
```

<details>
<summary>Full training command</summary>

```bash
accelerate launch --mixed_precision=bf16 --num_processes=1 \
  scripts/train_sdxl.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir data/dreambooth/dog \
  --output_dir output/dti-sdxl/dog \
  --placeholder_token "<dog>" \
  --initializer_token dog \
  --resolution 768 \
  --train_batch_size 4 \
  --max_train_steps 400 \
  --learning_rate 0.01 \
  --token_scale mean \
  --kappa 1e-4 \
  --decompose_scale true
```
</details>

### Train on SANA

```bash
python exps/ours_sana.py -g 0 -m sana1.5_1.6b
```

> Note: Our paper reports SANA with learning rate `0.02` and `1000` steps, but later experiments showed better performance with learning rate `0.01` and `500` steps. We recommend using the `0.01` + `500` combination for SANA.

### Train on Wan2.1-T2V-1.3B (image-as-video, 1 frame)

```bash
python exps/ours_wan.py -g 2 -m wan2.1_t2v_1.3b
```

> Wan uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. If you see
> `ValueError: Unrecognized model ...`, upgrade `transformers` and `diffusers`:
>
> `uv pip install --python .venv/bin/python --upgrade "git+https://github.com/huggingface/diffusers.git@main" "transformers>=4.48" "huggingface-hub>=0.34,<1.0"`

### Evaluate

```bash
python scripts/evaluate.py -e output/dti-sdxl
```

## Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-g` | GPU ID | `0` |
| `--instances` | Subject names | all |
| `--max_train_steps` | Training iterations | `400` |
| `--kappa` | vMF prior strength | `1e-4` |
| `--learning_rate` | Learning rate | `0.01` |
| `--token_scale` | Magnitude scale (`mean`, `max`, or float) | `mean` |

### Baselines

```bash
# Standard Textual Inversion
python exps/ti_sdxl.py -g 0

# DCO + DTI
python exps/ours_dco_sdxl.py -g 0
```

## Data Format

```
data/
├── dreambooth.json
└── dreambooth/
    └── subject_name/
        ├── 00.jpg
        ├── 01.jpg
        └── ...
```

JSON format:
```json
{
  "subject_name": {
    "path": "data/dreambooth/subject_name",
    "class": "dog",
    "initialization": "dog"
  }
}
```

Download DreamBooth dataset:
```bash
python scripts/download_datasets.py
```

## Project Structure

```
dti/
├── src/dti/       # Core implementation
├── scripts/       # Training & evaluation scripts
├── exps/          # Experiment launchers
└── data/          # Datasets and configs
```

## Citation

```bibtex
@article{kim2025directional,
  title={Directional Textual Inversion for Personalized Text-to-Image Generation},
  author={Kim, Kunhee and Park, NaHyeon and Hong, Kibeom and Shim, Hyunjung},
  journal={ICLR 2026},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
