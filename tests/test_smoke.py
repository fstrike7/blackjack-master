"""Basic smoke tests to ensure core modules import correctly."""

from blackjack import BlackjackAgent, BlackjackEnv, BlackjackTrainer


def test_environment_reset():
    env = BlackjackEnv(seed=42)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 3


def test_agent_forward_pass():
    agent = BlackjackAgent(state_size=3)
    state = BlackjackEnv(seed=42).get_state()
    action = agent.select_action(state, epsilon=0.0)
    assert action in (0, 1)


def test_trainer_instantiation():
    env = BlackjackEnv(seed=42)
    agent = BlackjackAgent(state_size=3)
    trainer = BlackjackTrainer(env, agent)
    assert trainer.config.default_episodes > 0
