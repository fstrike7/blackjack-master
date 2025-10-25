"""Blackjack environment providing basic game mechanics for training agents."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple, Union


Action = Union[int, str]


@dataclass
class StepResult:
    """Container for the environment `step` output."""

    state: Tuple[int, int, int]
    reward: float
    done: bool


class BlackjackEnv:
    """Simple Blackjack environment supporting hit/stand actions.

    The environment uses a finite deck (reshuffled when exhausted) and follows
    standard Blackjack rules where the dealer hits until reaching a value of 17
    or higher (treating soft 17 as 17).
    """

    ACTIONS: Tuple[str, str] = ("stand", "hit")

    def __init__(
        self, seed: int | None = None, natural_blackjack_bonus: float = 1.5
    ) -> None:
        """Initialise the environment.

        Args:
            seed: Optional seed for deterministic shuffling.
            natural_blackjack_bonus: Reward bonus for player Blackjack on the
                opening hand (win reward becomes this value).
        """
        self._rng = random.Random(seed)
        self.natural_blackjack_bonus = natural_blackjack_bonus
        self.deck: List[int] = []
        self.player_hand: List[int] = []
        self.dealer_hand: List[int] = []
        self.done = False
        self._initial_player_blackjack = False
        self.reset()

    # Public API -----------------------------------------------------------------
    def reset(self) -> Tuple[int, int, int]:
        """Reset the game and return the initial observable state.

        Returns:
            Tuple containing (player_total, dealer_showing, usable_ace_flag).
        """
        self.deck = self._generate_deck()
        self._rng.shuffle(self.deck)
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.done = False
        self._initial_player_blackjack = self._is_blackjack(self.player_hand)
        return self.get_state()

    def step(self, action: Action) -> StepResult:
        """Execute an action and advance the environment state.

        Args:
            action: Either an integer index (0=stand, 1=hit) or the literal
                strings "stand"/"hit".

        Returns:
            StepResult wrapping (state, reward, done).

        Raises:
            ValueError: If called after the episode ended or with an invalid action.
        """
        if self.done:
            raise ValueError(
                "Cannot call step() on a finished episode. Call reset() first."
            )

        action_index = self._normalise_action(action)

        if action_index == 1:  # hit
            self.player_hand.append(self._draw_card())
            player_total = self._hand_value(self.player_hand)
            if player_total > 21:
                self.done = True
                return StepResult(state=self.get_state(), reward=-1.0, done=True)
            return StepResult(state=self.get_state(), reward=0.0, done=False)

        # stand
        self.done = True
        self._play_dealer_hand()
        reward = self._compare_hands()
        return StepResult(state=self.get_state(), reward=reward, done=True)

    def get_state(self) -> Tuple[int, int, int]:
        """Return the current observable state for the agent."""
        player_total = self._hand_value(self.player_hand)
        dealer_showing = self._card_value(self.dealer_hand[0])
        usable_ace = int(self._has_usable_ace(self.player_hand))
        return player_total, dealer_showing, usable_ace

    # Internal helpers -----------------------------------------------------------
    def _normalise_action(self, action: Action) -> int:
        if isinstance(action, str):
            try:
                return self.ACTIONS.index(action.lower())
            except ValueError as exc:
                raise ValueError(f"Invalid action string: {action}") from exc
        if action not in (0, 1):
            raise ValueError(f"Invalid action index: {action}")
        return action

    def _generate_deck(self) -> List[int]:
        # Four suits, cards 1 (Ace) through 13 (King)
        return [card for card in range(1, 14) for _ in range(4)]

    def _draw_card(self) -> int:
        if not self.deck:
            # Reshuffle a fresh deck if exhausted.
            self.deck = self._generate_deck()
            self._rng.shuffle(self.deck)
        return self.deck.pop()

    def _card_value(self, card: int) -> int:
        return min(card, 10)

    def _hand_value(self, hand: Sequence[int]) -> int:
        total = sum(self._card_value(card) for card in hand)
        ace_count = sum(1 for card in hand if card == 1)
        # Upgrade aces to 11 while it keeps us <= 21.
        while ace_count > 0 and total + 10 <= 21:
            total += 10
            ace_count -= 1
        return total

    def _has_usable_ace(self, hand: Sequence[int]) -> bool:
        total = sum(self._card_value(card) for card in hand)
        return any(card == 1 for card in hand) and total + 10 <= 21

    def _play_dealer_hand(self) -> None:
        while True:
            total = self._hand_value(self.dealer_hand)
            # Dealer stands on soft 17.
            if total >= 17:
                break
            self.dealer_hand.append(self._draw_card())

    def _is_blackjack(self, hand: Sequence[int]) -> bool:
        return len(hand) == 2 and self._hand_value(hand) == 21

    def _compare_hands(self) -> float:
        player_total = self._hand_value(self.player_hand)
        dealer_total = self._hand_value(self.dealer_hand)

        if player_total > 21:
            return -1.0
        if dealer_total > 21:
            return 1.0

        if player_total > dealer_total:
            if self._initial_player_blackjack:
                return self.natural_blackjack_bonus
            return 1.0
        if player_total < dealer_total:
            return -1.0
        return 0.0
