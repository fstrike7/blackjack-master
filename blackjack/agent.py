"""PyTorch agent approximating state-action values for Blackjack."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


class BlackjackAgent(nn.Module):
    """Simple fully connected network for estimating Q-values."""

    def __init__(
        self,
        state_size: int,
        action_size: int = 2,
        hidden_layers: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        layers: list[nn.Module] = []
        input_dim = state_size
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    # PyTorch module API ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)

    # Agent helpers --------------------------------------------------------------
    def select_action(self, state: Iterable[float], epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        state_tensor = torch.tensor(list(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return Q-values for a batch of states."""
        return self.forward(state_batch.to(self.device))

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
    ) -> float:
        """Perform a single optimisation step on the provided batch."""
        self.optimizer.zero_grad()
        q_values = self.forward(states.to(self.device))
        action_q = q_values.gather(1, actions.to(self.device).unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(action_q, target_q.to(self.device))
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path_obj)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(Path(path), map_location=self.device)
        self.load_state_dict(state_dict)
