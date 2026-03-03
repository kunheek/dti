"""Per-step Text Alignment & Subject Fidelity plotting utilities.

This module provides reusable functions for loading per-checkpoint JSON files
from a folder and plotting text-alignment and subject-fidelity across steps.

Key feature: all functions accept an optional ``pattern`` glob string
(e.g. ``"score_simple-*-masked.json"`` or ``"score_simple-*-masked-rescale_mean.json"``)
so that different scoring variants living in the same folder can be
loaded and plotted independently.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_step_from_name(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def _to_numeric_values(v):
    if isinstance(v, dict):
        vals = [x for x in v.values() if isinstance(x, (int, float))]
        return np.array(vals, dtype=float) if vals else np.array([], dtype=float)
    if isinstance(v, (list, tuple, np.ndarray)):
        vals = [x for x in v if isinstance(x, (int, float))]
        return np.array(vals, dtype=float) if vals else np.array([], dtype=float)
    if isinstance(v, (int, float)):
        return np.array([float(v)], dtype=float)
    return np.array([], dtype=float)


def load_checkpoint_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return None

    text_vals = _to_numeric_values(data.get("text"))
    image_vals = _to_numeric_values(data.get("image"))

    if text_vals.size == 0:
        for k in ["text_alignment", "text_score", "caption_score"]:
            text_vals = _to_numeric_values(data.get(k))
            if text_vals.size:
                break
    if image_vals.size == 0:
        for k in ["subject_fidelity", "subject_score", "image_score", "fidelity"]:
            image_vals = _to_numeric_values(data.get(k))
            if image_vals.size:
                break

    if text_vals.size == 0 and image_vals.size == 0:
        return None

    step = _extract_step_from_name(Path(json_path).stem)

    return {
        "step": step,
        "text_alignment": float(np.nanmean(text_vals)) if text_vals.size else np.nan,
        "subject_fidelity": float(np.nanmean(image_vals))
        if image_vals.size
        else np.nan,
    }


def _collect_metrics_df(folder_path, *, pattern: str | None = None):
    """Collect per-checkpoint metrics from *folder_path*.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing JSON score files.
    pattern : str, optional
        A glob pattern to select only specific JSON files inside the folder.
        For example ``"score_simple-*-masked.json"`` or
        ``"score_simple-*-masked-rescale_mean.json"``.
        When *None* (the default), **all** ``*.json`` files are loaded.
    """
    p = Path(folder_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {folder_path}")

    if pattern is not None:
        files = sorted(p.glob(pattern))
    else:
        files = sorted([f for f in p.iterdir() if f.suffix.lower() == ".json"])

    if not files:
        raise ValueError(
            f"No JSON files found in {folder_path}"
            + (f" matching pattern '{pattern}'" if pattern else "")
        )

    rows = []
    for f in files:
        row = load_checkpoint_metrics(f)
        if row is None:
            continue
        row["checkpoint"] = f.stem
        rows.append(row)

    if not rows:
        raise ValueError("No plottable metrics found in JSON files.")

    all_df = pd.DataFrame(rows)
    if all_df["step"].isna().all():
        all_df["step"] = np.arange(1, len(all_df) + 1)

    all_df = all_df.sort_values("step", na_position="last").reset_index(drop=True)
    return all_df


# ---------------------------------------------------------------------------
# print summary
# ---------------------------------------------------------------------------

def print_scores(folder_path, *, pattern: str | None = None):
    """Print mean text-alignment and subject-fidelity for each JSON file.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing JSON score files.
    pattern : str, optional
        Glob pattern to filter files (e.g. ``"score_simple-*-masked.json"``).

    Examples
    --------
    >>> print_scores("../outputs/dti-sana1.5_1.6b-camera/",
    ...              pattern="score_simple-*-masked.json")
    """
    df = _collect_metrics_df(folder_path, pattern=pattern)
    print(f"{'checkpoint':<50} {'text':>8} {'image':>8}")
    print("-" * 68)
    for _, row in df.iterrows():
        i = f"{row['subject_fidelity']:.3f}" if pd.notna(row["subject_fidelity"]) else "   N/A"
        t = f"{row['text_alignment']:.3f}" if pd.notna(row["text_alignment"]) else "   N/A"
        print(f"{row['checkpoint']:<50} {i:>8} {t:>8}")
    print("-" * 68)
    # print(f"{'mean':<50} {df['text_alignment'].mean():>8.3f} {df['subject_fidelity'].mean():>8.3f}")


# ---------------------------------------------------------------------------
# single-experiment plots
# ---------------------------------------------------------------------------

def plot_scores(folder_path, *, pattern=None, figsize=(12, 5), save_path=None):
    all_df = _collect_metrics_df(folder_path, pattern=pattern)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.lineplot(data=all_df, x="step", y="text_alignment", marker="o", ax=axes[0])
    axes[0].set_title("Text Alignment")
    axes[0].set_xlabel("Checkpoint Step")
    axes[0].set_ylabel("Score")

    sns.lineplot(data=all_df, x="step", y="subject_fidelity", marker="o", ax=axes[1])
    axes[1].set_title("Subject Fidelity")
    axes[1].set_xlabel("Checkpoint Step")
    axes[1].set_ylabel("Score")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved figure to", save_path)
    plt.show()


def plot_xy_trajectory(
    folder_path,
    *,
    pattern=None,
    figsize=(7, 6),
    color="tab:blue",
    save_path=None,
):
    """
    x-axis: subject_fidelity (image)
    y-axis: text_alignment (text)
    Connect checkpoints in step order and use increasing opacity for later
    checkpoints.
    """
    df = _collect_metrics_df(folder_path, pattern=pattern)

    fig, ax = plt.subplots(figsize=figsize)

    x = df["subject_fidelity"].to_numpy()
    y = df["text_alignment"].to_numpy()
    steps = df["step"].to_numpy()
    n = len(df)

    ax.plot(x, y, color=color, alpha=0.35, linewidth=1.8)

    min_alpha, max_alpha = 0.25, 1.0
    alphas = np.linspace(min_alpha, max_alpha, n)

    for i in range(n):
        ax.scatter(
            x[i], y[i], s=55, color=color, alpha=float(alphas[i]), edgecolor="none"
        )
        ax.text(
            x[i],
            y[i],
            str(int(steps[i])) if pd.notna(steps[i]) else str(i + 1),
            fontsize=8,
            alpha=0.8,
        )
        if i > 0:
            ax.plot(
                x[i - 1 : i + 1],
                y[i - 1 : i + 1],
                color=color,
                alpha=float(alphas[i]),
                linewidth=2.2,
            )

    ax.set_title("Checkpoint Trajectory (Image vs Text)")
    ax.set_xlabel("Image (Subject Fidelity)")
    ax.set_ylabel("Text (Text Alignment)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved figure to", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# multi-experiment comparison plots
# ---------------------------------------------------------------------------

def compare_scores(
    *folders,
    labels=None,
    pattern=None,
    figsize=(12, 5),
    save_path=None,
):
    """
    Plot text-alignment and subject-fidelity for N experiments on the same
    axes.

    Args:
        *folders: One or more folder paths (str or Path).
        labels:   Optional list of legend labels (same length as folders).
        pattern:  Glob pattern to select specific JSON files inside each
                  folder (e.g. ``"score_simple-*-masked.json"``).
        figsize:  Figure size.
        save_path: If given, save the figure here.

    Examples:
        compare_scores("outputs/a/", "outputs/b/",
                        pattern="score_simple-*-masked.json")
    """
    if not folders:
        raise ValueError("Provide at least one folder path.")
    if labels is None:
        labels = [Path(f).name for f in folders]
    if len(labels) != len(folders):
        raise ValueError("len(labels) must match the number of folders.")

    dfs = []
    for folder, label in zip(folders, labels):
        df = _collect_metrics_df(folder, pattern=pattern)
        df["experiment"] = label
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.lineplot(
        data=combined,
        x="step",
        y="text_alignment",
        hue="experiment",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Text Alignment")
    axes[0].set_xlabel("Checkpoint Step")
    axes[0].set_ylabel("Score")

    sns.lineplot(
        data=combined,
        x="step",
        y="subject_fidelity",
        hue="experiment",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Subject Fidelity")
    axes[1].set_xlabel("Checkpoint Step")
    axes[1].set_ylabel("Score")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved figure to", save_path)
    plt.show()


def compare_trajectories(
    *folders,
    labels=None,
    colors=None,
    pattern=None,
    figsize=(7, 6),
    save_path=None,
):
    """
    Overlay checkpoint trajectories for N experiments on a single x-y plot.
    x-axis: subject_fidelity, y-axis: text_alignment.

    Args:
        *folders:  One or more folder paths (str or Path).
        labels:    Optional list of legend labels (same length as folders).
        colors:    Optional list of colors (same length as folders).
        pattern:   Glob pattern to select specific JSON files inside each
                   folder (e.g. ``"score_simple-*-masked.json"``).
        figsize:   Figure size.
        save_path: If given, save the figure here.

    Examples:
        compare_trajectories(
            "outputs/a/", "outputs/b/",
            pattern="score_simple-*-masked-rescale_mean.json",
        )
    """
    if not folders:
        raise ValueError("Provide at least one folder path.")
    if labels is None:
        labels = [Path(f).name for f in folders]
    if len(labels) != len(folders):
        raise ValueError("len(labels) must match the number of folders.")

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if colors is None:
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(folders))]
    if len(colors) != len(folders):
        raise ValueError("len(colors) must match the number of folders.")

    fig, ax = plt.subplots(figsize=figsize)

    for folder, label, color in zip(folders, labels, colors):
        df = _collect_metrics_df(folder, pattern=pattern)
        x = df["subject_fidelity"].to_numpy()
        y = df["text_alignment"].to_numpy()
        steps = df["step"].to_numpy()
        n = len(df)

        ax.plot(x, y, color=color, alpha=0.35, linewidth=1.8, label=label)

        alphas = np.linspace(0.25, 1.0, n)
        for i in range(n):
            ax.scatter(
                x[i], y[i], s=55, color=color, alpha=float(alphas[i]), edgecolor="none"
            )
            ax.text(
                x[i],
                y[i],
                str(int(steps[i])) if pd.notna(steps[i]) else str(i + 1),
                fontsize=8,
                alpha=0.8,
            )
            if i > 0:
                ax.plot(
                    x[i - 1 : i + 1],
                    y[i - 1 : i + 1],
                    color=color,
                    alpha=float(alphas[i]),
                    linewidth=2.2,
                )

    ax.set_title("Checkpoint Trajectory Comparison (Image vs Text)")
    ax.set_xlabel("Image (Subject Fidelity)")
    ax.set_ylabel("Text (Text Alignment)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved figure to", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# single-file plot
# ---------------------------------------------------------------------------

def plot_single_file(json_path, *, figsize=(14, 6), save_path=None):
    """Plot per-subject scores from a single JSON score file.

    Shows two horizontal bar charts side-by-side:
      - **Text Alignment** (left)
      - **Subject Fidelity** (right)

    A dashed vertical line marks the mean score in each panel.

    Parameters
    ----------
    json_path : str or Path
        Path to a specific JSON score file such as
        ``"score_simple-100-masked.json"``.
    figsize : tuple, optional
        Figure size.
    save_path : str or Path, optional
        If given, save the figure to this path.

    Examples
    --------
    >>> plot_single_file("outputs/dti-sana1.5_1.6b-camera/score_simple-100-masked.json")
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(p, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    # Resolve text / image dicts ------------------------------------------
    text_dict = data.get("text") or {}
    image_dict = data.get("image") or {}

    # Fallback keys
    if not text_dict:
        for k in ["text_alignment", "text_score", "caption_score"]:
            if isinstance(data.get(k), dict):
                text_dict = data[k]
                break
    if not image_dict:
        for k in ["subject_fidelity", "subject_score", "image_score", "fidelity"]:
            if isinstance(data.get(k), dict):
                image_dict = data[k]
                break

    if not text_dict and not image_dict:
        raise ValueError(f"No plottable data found in {json_path}")

    # Build a merged subject list (sorted) --------------------------------
    subjects = sorted(set(list(text_dict.keys()) + list(image_dict.keys())))

    text_scores = [text_dict.get(s, np.nan) for s in subjects]
    image_scores = [image_dict.get(s, np.nan) for s in subjects]

    n_panels = (1 if text_dict else 0) + (1 if image_dict else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if text_dict:
        ax = axes[panel_idx]
        y_pos = np.arange(len(subjects))
        ax.barh(y_pos, text_scores, color="steelblue", edgecolor="none", height=0.7)
        mean_val = float(np.nanmean(text_scores))
        ax.axvline(mean_val, color="tomato", linestyle="--", linewidth=1.2,
                   label=f"mean = {mean_val:.4f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subjects, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title("Text Alignment")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        panel_idx += 1

    if image_dict:
        ax = axes[panel_idx]
        y_pos = np.arange(len(subjects))
        ax.barh(y_pos, image_scores, color="darkorange", edgecolor="none", height=0.7)
        mean_val = float(np.nanmean(image_scores))
        ax.axvline(mean_val, color="tomato", linestyle="--", linewidth=1.2,
                   label=f"mean = {mean_val:.4f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subjects, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title("Subject Fidelity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(p.name, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved figure to", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# quick examples
# ---------------------------------------------------------------------------
# from plot import plot_single_file, compare_trajectories
#
# # Plot a specific JSON file:
# plot_single_file("../outputs/dti-sana1.5_1.6b-camera/score_simple-100-masked.json")
#
# # Plot only "masked" scores:
# compare_trajectories(
#     "../outputs/dti-sana1.5_1.6b-camera/",
#     "../outputs/ti-sana1.5_1.6b-camera/",
#     "../outputs/xinit-sana1.5_1.6b-camera/",
#     pattern="score_simple-*-masked.json",
# )
#
# # Plot only "masked-rescale_mean" scores:
# compare_trajectories(
#     "../outputs/dti-sana1.5_1.6b-camera/",
#     "../outputs/ti-sana1.5_1.6b-camera/",
#     "../outputs/xinit-sana1.5_1.6b-camera/",
#     pattern="score_simple-*-masked-rescale_mean.json",
# )
