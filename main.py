"""Command-line interface for the Blackjack Learner project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from blackjack import BlackjackAgent, BlackjackEnv, BlackjackTrainer, TrainerConfig
from blackjack.utils import configure_logging, ensure_results_file
from visualization.graphs import plot_reward_curve


STATE_SIZE = 3
RESULTS_PATH = Path("data/results.csv")
CHECKPOINT_DIR = Path("data/checkpoints")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blackjack Learner CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Blackjack agent")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    train_parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Exploration decay factor")

    viz_parser = subparsers.add_parser("visualize", help="Visualise training metrics")
    viz_parser.add_argument("--save", type=Path, default=None, help="Optional path to save the plot image")
    viz_parser.add_argument("--no-show", action="store_true", help="Generate plot without displaying it")

    play_parser = subparsers.add_parser("play", help="Play a Blackjack round against the dealer")
    play_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser


def run_train(args: argparse.Namespace) -> None:
    logger = configure_logging()
    logger.info("Starting training for %d episodes", args.episodes)

    env = BlackjackEnv(seed=args.seed)
    agent = BlackjackAgent(state_size=STATE_SIZE, learning_rate=args.learning_rate)

    config = TrainerConfig(
        default_episodes=args.episodes,
        epsilon_decay=args.epsilon_decay,
        results_path=RESULTS_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    trainer = BlackjackTrainer(env, agent, config=config)
    trainer.train(episodes=args.episodes)
    logger.info("Training finished. Results stored in %s", RESULTS_PATH)


def run_visualize(args: argparse.Namespace) -> None:
    show_plot = not args.no_show
    plot_reward_curve(csv_path=RESULTS_PATH, show=show_plot, save_path=args.save)


def run_play(args: argparse.Namespace) -> None:
    env = BlackjackEnv(seed=args.seed)
    ensure_results_file(RESULTS_PATH)

    print("Starting a new Blackjack round. Enter 'hit' or 'stand'. Type 'quit' to exit.")
    state = env.reset()
    _print_state(state, hide_dealer=False)

    while True:
        user_input = input("Action (hit/stand/quit): ").strip().lower()
        if user_input in {"quit", "q", "exit"}:
            print("Exiting game.")
            return
        if user_input not in {"hit", "stand"}:
            print("Invalid action. Please enter 'hit' or 'stand'.")
            continue

        step_result = env.step(user_input)
        _print_state(step_result.state, hide_dealer=False)

        if step_result.done:
            if step_result.reward > 0:
                print("You win!")
            elif step_result.reward < 0:
                print("You lose!")
            else:
                print("It's a tie.")
            break


def _print_state(state: tuple[int, int, int], hide_dealer: bool = True) -> None:
    player_total, dealer_showing, usable_ace = state
    dealer_info = "Hidden" if hide_dealer else str(dealer_showing)
    print(
        f"Player total: {player_total} | Dealer showing: {dealer_info} | "
        f"Usable ace: {'Yes' if usable_ace else 'No'}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        run_train(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "play":
        run_play(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
