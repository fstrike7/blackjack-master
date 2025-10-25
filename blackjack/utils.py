"""Utility helpers for logging, persistence, and result handling."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


RESULTS_HEADER = [
    "episode",
    "total_reward",
    "decisions",
    "wins",
    "losses",
    "ties",
]


@dataclass
class EpisodeResult:
    """Structured record of a training episode outcome."""

    episode: int
    total_reward: float
    decisions: int
    wins: int
    losses: int
    ties: int

    def as_row(self) -> List[str]:
        return [str(value) for value in asdict(self).values()]


def ensure_results_file(path: str | Path) -> Path:
    """Ensure the CSV file exists with the correct header."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not path_obj.exists():
        with path_obj.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(RESULTS_HEADER)
    return path_obj


def append_episode_results(path: str | Path, result: EpisodeResult) -> None:
    """Append a single episode result to the CSV file."""
    path_obj = ensure_results_file(path)
    with path_obj.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(result.as_row())


def load_episode_results(path: str | Path) -> List[EpisodeResult]:
    """Load recorded episode results."""
    path_obj = ensure_results_file(path)
    results: List[EpisodeResult] = []
    with path_obj.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            results.append(
                EpisodeResult(
                    episode=int(row["episode"]),
                    total_reward=float(row["total_reward"]),
                    decisions=int(row["decisions"]),
                    wins=int(row["wins"]),
                    losses=int(row["losses"]),
                    ties=int(row["ties"]),
                )
            )
    return results


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger("blackjack")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
