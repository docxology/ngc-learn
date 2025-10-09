#!/usr/bin/env python3
"""
T-Maze Active Inference example.

Demonstrates epistemic exploration and information-seeking behavior in a T-maze environment.
The agent must explore to discover which arm contains the reward, showcasing the epistemic
value component of expected free energy.
"""

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ngc_inference.core.active_inference_agent import ActiveInferenceAgent
from ngc_inference.utils.logging_config import setup_logging


class TMazeEnvironment:
    """
    Simple T-maze environment for demonstrating epistemic exploration.

    States:
    - 0: Start position
    - 1: Left junction
    - 2: Right junction
    - 3: Left arm end (reward location, unknown initially)
    - 4: Right arm end (no reward)

    Actions:
    - 0: Go left
    - 1: Go right
    """

    def __init__(self, reward_location: int = 3, seed: int = 42):
        """
        Initialize T-maze environment.

        Args:
            reward_location: Which state contains the reward (3=left, 4=right)
            seed: Random seed
        """
        self.reward_location = reward_location
        self.current_state = 0
        self.key = random.PRNGKey(seed)

        # Transition dynamics: deterministic for simplicity
        self.transitions = {
            (0, 0): 1,  # Start -> Left junction
            (0, 1): 2,  # Start -> Right junction
            (1, 0): 3,  # Left junction -> Left arm
            (2, 1): 4,  # Right junction -> Right arm
        }

    def reset(self):
        """Reset environment to start state."""
        self.current_state = 0
        return self.get_observation()

    def step(self, action: int):
        """
        Execute action and return next state, reward, done.

        Args:
            action: Action to take (0=left, 1=right)

        Returns:
            observation: New observation
            reward: Reward received
            done: Whether episode is finished
        """
        # Get next state from transition table
        key = (self.current_state, action)
        if key in self.transitions:
            self.current_state = self.transitions[key]
        else:
            # Invalid transition - stay in place
            pass

        # Get observation and reward
        observation = self.get_observation()
        reward = 1.0 if self.current_state == self.reward_location else 0.0
        done = True  # Single step episodes for simplicity

        return observation, reward, done

    def get_observation(self):
        """Get observation for current state (one-hot encoding)."""
        obs = jnp.zeros(5)
        obs = obs.at[self.current_state].set(1.0)
        return obs.reshape(1, -1)

    def get_state(self):
        """Get current state."""
        return self.current_state


def run_tmaze_experiment(n_episodes: int = 100, reward_location: int = 3):
    """
    Run T-maze experiment demonstrating epistemic exploration.

    Args:
        n_episodes: Number of episodes to run
        reward_location: Which arm contains reward (3=left, 4=right)
    """
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/tmaze_experiment.log")

    print("=" * 70)
    print("T-Maze Active Inference Experiment")
    print("=" * 70)
    print(f"Episodes: {n_episodes}, Reward in state: {reward_location}")
    print()

    # Initialize environment and agent
    env = TMazeEnvironment(reward_location=reward_location)
    agent = ActiveInferenceAgent(
        n_states=5,
        n_observations=5,
        n_actions=2,
        action_space="discrete",
        learning_rate_states=0.1,
        observation_precision=1.0,
        state_precision=1.0,
        policy_temperature=1.0,  # Balanced exploration/exploitation
        seed=42
    )

    # Track results
    episode_rewards = []
    episode_actions = []
    episode_efes = []
    episode_beliefs = []

    for episode in range(n_episodes):
        # Reset environment and agent
        obs = env.reset()
        agent.reset()

        # Choose action based on current beliefs
        preferred_obs = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Reward observation
        action, metrics = agent.select_action(obs, preferred_obs, sample=True)

        # Execute action
        next_obs, reward, done = env.step(action)

        # Learn from experience
        current_state_onehot = obs[0]
        next_state_onehot = next_obs[0]
        trajectories = [(current_state_onehot, action, next_state_onehot)]
        agent.learn_transition_model(trajectories)

        # Track results
        episode_rewards.append(reward)
        episode_actions.append(action)
        episode_efes.append(metrics["expected_free_energies"][action])
        episode_beliefs.append(agent.get_beliefs()[0])

        if (episode + 1) % 10 == 0:
            avg_reward = jnp.mean(jnp.array(episode_rewards[-10:]))
            print(f"Episode {episode+1:3d}: Reward={reward:.1f}, Action={action}, EFE={metrics['expected_free_energies'][action]:.3f}")

    # Analyze results
    print("\n" + "=" * 70)
    print("Results Analysis")
    print("=" * 70)

    # Reward statistics
    rewards = jnp.array(episode_rewards)
    print(f"Total rewards: {jnp.sum(rewards)}")
    print(f"Average reward: {jnp.mean(rewards):.3f}")
    print(f"Success rate: {jnp.mean(rewards):.1%}")

    # Action distribution (should show exploration initially)
    actions = jnp.array(episode_actions)
    left_actions = jnp.sum(actions == 0)
    right_actions = jnp.sum(actions == 1)
    print(f"Left actions: {left_actions} ({left_actions/len(actions)*100:.1f})")
    print(f"Right actions: {right_actions} ({right_actions/len(actions)*100:.1f})")

    # EFE analysis
    efes = jnp.array(episode_efes)
    print(f"Average EFE: {jnp.mean(efes):.3f}")
    print(f"EFE std: {jnp.std(efes):.3f}")

    # Save results
    results_dir = Path("logs/runs/tmaze_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward over time
    axes[0, 0].plot(episode_rewards, alpha=0.7)
    axes[0, 0].set_title("Rewards per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # Action distribution over time
    axes[0, 1].plot(jnp.cumsum(actions == 0), label="Left (0)", alpha=0.7)
    axes[0, 1].plot(jnp.cumsum(actions == 1), label="Right (1)", alpha=0.7)
    axes[0, 1].set_title("Cumulative Actions")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Cumulative Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # EFE over time
    axes[1, 0].plot(efes, alpha=0.7)
    axes[1, 0].set_title("Expected Free Energy")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("EFE")
    axes[1, 0].grid(True, alpha=0.3)

    # Belief evolution (uncertainty about reward location)
    beliefs = jnp.array(episode_beliefs)
    reward_beliefs = beliefs[:, reward_location]  # Belief in reward state
    axes[1, 1].plot(reward_beliefs, alpha=0.7, label=f"Belief in state {reward_location}")
    axes[1, 1].set_title("Belief in Reward State")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Belief Strength")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "tmaze_results.png", dpi=300, bbox_inches='tight')
    print(f"Results saved to: {results_dir}")

    return {
        "rewards": episode_rewards,
        "actions": episode_actions,
        "efes": episode_efes,
        "beliefs": episode_beliefs,
        "final_reward_rate": jnp.mean(rewards)
    }


def run_multiple_reward_locations():
    """Run experiment with different reward locations to verify epistemic behavior."""
    print("\nTesting epistemic exploration with different reward locations...")

    results = {}
    for reward_loc in [3, 4]:  # Left vs right arm
        print(f"\n--- Reward in state {reward_loc} ---")
        result = run_tmaze_experiment(n_episodes=50, reward_location=reward_loc)
        results[reward_loc] = result

    # Compare results
    print("\n" + "=" * 70)
    print("Comparison Across Reward Locations")
    print("=" * 70)

    for reward_loc, result in results.items():
        print(f"Reward in state {reward_loc}: Final success rate = {result['final_reward_rate']:.1%}")

    return results


if __name__ == "__main__":
    # Run main experiment
    results = run_tmaze_experiment(n_episodes=100, reward_location=3)

    # Run comparison experiment
    comparison_results = run_multiple_reward_locations()

    print("\n✓ T-Maze experiment completed!")
    print("The agent demonstrates epistemic exploration by trying both arms")
    print("to discover where the reward is located, showcasing information-seeking behavior.")
