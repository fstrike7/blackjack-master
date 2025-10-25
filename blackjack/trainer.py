"""Training loop orchestration for the Blackjack agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import torch

from .agent import BlackjackAgent
from .environment import BlackjackEnv
from .utils import (
    EpisodeResult,
    append_episode_results,
    configure_logging,
    ensure_results_file,
)


@dataclass
class TrainerConfig:
    """Configuration parameters controlling the training loop."""

    default_episodes: int = 1000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    checkpoint_interval: int = 100
    results_path: Path = field(default_factory=lambda: Path("data/results.csv"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/checkpoints"))


class BlackjackTrainer:
    """Run training episodes for the Blackjack agent."""

    def __init__(
        self,
        env: BlackjackEnv,
        agent: BlackjackAgent,
        config: TrainerConfig | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or TrainerConfig()
        self.logger = configure_logging()
        ensure_results_file(self.config.results_path)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, episodes: int | None = None) -> List[EpisodeResult]:
        """Train the agent for the requested number of episodes."""
        num_episodes = episodes or self.config.default_episodes
        if num_episodes <= 0:
            raise ValueError("Number of episodes must be a positive integer.")

        results: List[EpisodeResult] = []
        epsilon = self.config.epsilon_start

        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            decisions = 0
            wins = losses = ties = 0

            while not done:
                action = self.agent.select_action(state, epsilon=epsilon)
                step_result = self.env.step(action)
                next_state = step_result.state

                reward = step_result.reward
                total_reward += reward
                decisions += 1
                done = step_result.done

                if done:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        ties += 1

                self._optimise_model(state, action, reward, next_state, done)
                state = next_state

            result = EpisodeResult(
                episode=episode,
                total_reward=total_reward,
                decisions=decisions,
                wins=wins,
                losses=losses,
                ties=ties,
            )
            append_episode_results(self.config.results_path, result)
            results.append(result)

            if episode % self.config.checkpoint_interval == 0:
                checkpoint_path = self.config.checkpoint_dir / f"agent_ep{episode}.pt"
                self.agent.save(checkpoint_path)
                self.logger.info("Saved checkpoint to %s", checkpoint_path)

            epsilon = max(self.config.epsilon_end, epsilon * self.config.epsilon_decay)
            if episode % 100 == 0 or episode == 1:
                self.logger.info(
                    "Episode %d/%d - reward=%.2f epsilon=%.3f",
                    episode,
                    num_episodes,
                    total_reward,
                    epsilon,
                )

        return results

    def _optimise_model(
        self,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        state_tensor = torch.tensor(list(state), dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(list(next_state), dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)

        with torch.no_grad():
            next_q = self.agent.predict(next_state_tensor)
            max_next_q = torch.max(next_q, dim=1).values.item()

        target_value = reward if done else reward + self.config.gamma * max_next_q
        target_tensor = torch.tensor([target_value], dtype=torch.float32)

        loss = self.agent.update(state_tensor, action_tensor, target_tensor)
        self.logger.debug("Loss: %.6f", loss)
