from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_METRICS = ["additive", "Stability_z", "CoreQuality_z", "Solubility_z"]


def plot_history(csv_path: str, output_dir: str = ".", metrics: Iterable[str] | None = None) -> None:
    """Plot per-generation averages from a history CSV.

    Parameters
    ----------
    csv_path : str
        Path to the ``history.csv`` file produced by ``EvoSage.main``.
    output_dir : str, optional
        Where to write the plots. The directory is created if needed.
    metrics : Iterable[str] | None, optional
        Metrics to plot. Defaults to ``DEFAULT_METRICS``.
    """
    df = pd.read_csv(csv_path)
    metrics = list(metrics or DEFAULT_METRICS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_means = df.groupby("gen")[metrics].mean()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3 * len(metrics)))
    if not isinstance(axes, Iterable):
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.lineplot(x=gen_means.index, y=gen_means[metric], marker="o", ax=ax)
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric)
        ax.set_title(f"Mean {metric} per generation")
    plt.tight_layout()
    plt.savefig(out / "per_generation_metrics.png", dpi=300)
    plt.close(fig)


def plot_final_scatter(csv_path: str, output_dir: str = ".", metrics: Iterable[str] | None = None) -> None:
    """Plot a pairwise scatter matrix for the final generation.

    Parameters
    ----------
    csv_path : str
        Path to the ``history.csv`` file produced by ``EvoSage.main``.
    output_dir : str, optional
        Where to write the plot. The directory is created if needed.
    metrics : Iterable[str] | None, optional
        Metrics to include. Defaults to ``DEFAULT_METRICS[1:]``.
    """
    df = pd.read_csv(csv_path)
    metrics = list(metrics or DEFAULT_METRICS[1:])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    last_gen = df["gen"].max()
    subset = df[df["gen"] == last_gen]

    sns.pairplot(subset, vars=metrics, height=3)
    plt.suptitle(f"Metrics for generation {last_gen}", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "final_generation_pairplot.png", dpi=300)
    plt.close()
