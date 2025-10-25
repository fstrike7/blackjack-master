"""Visualisation utilities for Blackjack training metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from blackjack.utils import ensure_results_file


def plot_reward_curve(
    csv_path: str | Path = "data/results.csv",
    show: bool = True,
    save_path: str | Path | None = None,
) -> Path | None:
    """Plot the rolling reward evolution from the results CSV.

    Args:
        csv_path: Location of the training results CSV file.
        show: Whether to display the plot interactively.
        save_path: Optional path to save the generated plot image.

    Returns:
        Path to the saved figure if `save_path` is provided and the plot is saved.
    """
    csv_file = ensure_results_file(csv_path)
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError("Results CSV is empty. Run training before visualising.")

    df["rolling_reward"] = df["total_reward"].rolling(window=50, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["total_reward"], label="Episode Reward", alpha=0.4)
    plt.plot(df["episode"], df["rolling_reward"], label="Rolling Avg (window=50)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Blackjack Learner - Reward Progression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    saved_path_obj: Path | None = None
    if save_path:
        saved_path_obj = Path(save_path)
        saved_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saved_path_obj)

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path_obj
