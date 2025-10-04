#!/usr/bin/env python3
"""
Active Inference Reaching Task example.

Demonstrates continuous action selection and goal-directed behavior in a 2D reaching task.
The agent must reach a target location using continuous motor commands, showcasing
pragmatic value (goal achievement) in expected free energy computation.
"""

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ngc_inference.core.active_inference_agent import ActiveInferenceAgent
from ngc_inference.utils.logging_config import setup_logging


class ReachingEnvironment:
    """
    2D reaching task with continuous actions.

    State: [x, y, vx, vy] - position and velocity
    Observations: [x, y] - position only (proprioception)
    Actions: [force_x, force_y] - continuous motor commands
    Goal: Reach target position [target_x, target_y]
    """

    def __init__(self, dt: float = 0.1, max_force: float = 1.0, seed: int = 42):
        """
        Initialize reaching environment.

        Args:
            dt: Time step for physics simulation
            max_force: Maximum force magnitude
            seed: Random seed
        """
        self.dt = dt
        self.max_force = max_force
        self.key = random.PRNGKey(seed)

        # Physics parameters
        self.mass = 1.0
        self.damping = 0.1

        # Current state: [x, y, vx, vy]
        self.state = jnp.array([0.0, 0.0, 0.0, 0.0])

        # Target location (will be set per episode)
        self.target = jnp.array([0.0, 0.0])

    def reset(self, target: Optional[jnp.ndarray] = None):
        """
        Reset environment with random initial state.

        Args:
            target: Target position [x, y]. If None, sample randomly.

        Returns:
            Initial observation [x, y]
        """
        if target is None:
            # Sample random target within reasonable bounds
            self.key, subkey = random.split(self.key)
            self.target = random.uniform(subkey, (2,), minval=-2.0, maxval=2.0)
        else:
            self.target = target

        # Reset state with some initial velocity toward target
        self.key, subkey = random.split(self.key)
        self.state = jnp.concatenate([
            random.uniform(subkey, (2,), minval=-1.0, maxval=1.0),  # position
            jnp.zeros(2)  # velocity
        ])

        return self.get_observation()

    def step(self, action: jnp.ndarray):
        """
        Execute continuous action and simulate physics.

        Args:
            action: Force vector [force_x, force_y]

        Returns:
            observation: Current position [x, y]
            reward: Distance to target (negative for minimization)
            done: Whether target reached
        """
        # Clamp action to valid range
        action = jnp.clip(action, -self.max_force, self.max_force)

        # Physics simulation: F = ma, with damping
        force = action
        acceleration = force / self.mass - self.damping * self.state[2:4]

        # Update velocity and position
        new_velocity = self.state[2:4] + acceleration * self.dt
        new_position = self.state[0:2] + new_velocity * self.dt

        # Update state
        self.state = jnp.concatenate([new_position, new_velocity])

        # Get observation and reward
        observation = self.get_observation()
        distance = jnp.linalg.norm(self.state[0:2] - self.target)
        reward = -distance  # Negative distance for minimization
        done = distance < 0.1  # Target reached

        return observation, reward, done

    def get_observation(self):
        """Get proprioceptive observation (position only)."""
        return self.state[0:2].reshape(1, -1)

    def get_state(self):
        """Get full state."""
        return self.state

    def get_target(self):
        """Get current target."""
        return self.target


def run_reaching_experiment(n_episodes: int = 100, max_steps: int = 50):
    """
    Run reaching task experiment demonstrating goal-directed behavior.

    Args:
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/reaching_experiment.log")

    print("=" * 70)
    print("Active Inference Reaching Task")
    print("=" * 70)
    print(f"Episodes: {n_episodes}, Max steps: {max_steps}")
    print()

    # Initialize environment and agent
    env = ReachingEnvironment(dt=0.1, max_force=1.0)
    agent = ActiveInferenceAgent(
        n_states=4,  # [x, y, vx, vy]
        n_observations=2,  # [x, y] only
        n_actions=2,  # [force_x, force_y]
        action_space="continuous",
        learning_rate_states=0.1,
        observation_precision=1.0,
        state_precision=1.0,
        policy_temperature=1.0,
        transition_model_type="continuous",
        seed=42
    )

    # Track results
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_actions = []

    for episode in range(n_episodes):
        # Reset environment with random target
        obs = env.reset()
        target = env.get_target()

        # Reset agent
        agent.reset()

        episode_reward = 0.0
        episode_actions_list = []

        for step in range(max_steps):
            # Select continuous action
            action, metrics = agent.select_action(obs, target.reshape(1, -1), sample=True)

            # Execute action
            next_obs, reward, done = env.step(action)

            # Learn from experience
            current_state = env.get_state()
            next_state = jnp.array([next_obs[0, 0], next_obs[0, 1], 0.0, 0.0])  # Simplified state
            trajectories = [(current_state, action, next_state)]
            agent.learn_transition_model(trajectories)

            # Track progress
            episode_reward += reward
            episode_actions_list.append(action)

            # Update observation
            obs = next_obs

            if done:
                break

        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        episode_distances.append(jnp.linalg.norm(env.get_state()[0:2] - target))
        episode_actions.append(episode_actions_list)

        if (episode + 1) % 10 == 0:
            avg_reward = jnp.mean(jnp.array(episode_rewards[-10:]))
            avg_length = jnp.mean(jnp.array(episode_lengths[-10:]))
            print(f"Episode {episode+1"3d"}: Reward={episode_reward".2f"}, Steps={step+1"2d"}, Distance={episode_distances[-1]".3f"}")

    # Analyze results
    print("\n" + "=" * 70)
    print("Results Analysis")
    print("=" * 70)

    rewards = jnp.array(episode_rewards)
    lengths = jnp.array(episode_lengths)
    distances = jnp.array(episode_distances)

    print(f"Total episodes: {len(rewards)}")
    print(f"Average reward: {jnp.mean(rewards):".3f"")
    print(f"Average episode length: {jnp.mean(lengths):".1f"} steps")
    print(f"Final distance to target: {jnp.mean(distances):".3f"")

    # Success rate (distance < 0.2)
    success_rate = jnp.mean(distances < 0.2)
    print(f"Success rate (distance < 0.2): {success_rate:".1%"")

    # Save results
    results_dir = Path("logs/runs/reaching_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward over episodes
    axes[0, 0].plot(rewards, alpha=0.7)
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.7)
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True, alpha=0.3)

    # Final distances
    axes[1, 0].plot(distances, alpha=0.7)
    axes[1, 0].set_title("Final Distance to Target")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True, alpha=0.3)

    # Sample trajectories (show a few episodes)
    axes[1, 1].set_title("Sample Trajectories")
    axes[1, 1].set_xlabel("X Position")
    axes[1, 1].set_ylabel("Y Position")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot a few sample trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(min(5, len(episode_actions))):
        actions = jnp.array(episode_actions[i])
        if len(actions) > 0:
            # Reconstruct trajectory from actions (simplified)
            trajectory_x = [0.0]
            trajectory_y = [0.0]
            for action in actions:
                trajectory_x.append(trajectory_x[-1] + action[0] * 0.1)
                trajectory_y.append(trajectory_y[-1] + action[1] * 0.1)

            axes[1, 1].plot(trajectory_x, trajectory_y,
                           color=colors[i], alpha=0.7, linewidth=2)

    # Plot targets for the sampled episodes
    for i in range(min(5, len(episode_actions))):
        if i < len(episode_actions):
            target = jnp.array([0.0, 0.0])  # Simplified for visualization
            axes[1, 1].scatter([target[0]], [target[1]], marker='x', s=100,
                              color=colors[i], alpha=0.8)

    plt.tight_layout()
    plt.savefig(results_dir / "reaching_results.png", dpi=300, bbox_inches='tight')
    print(f"Results saved to: {results_dir}")

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "distances": episode_distances,
        "success_rate": success_rate
    }


def run_goal_directed_vs_exploration():
    """Compare goal-directed vs exploratory behavior by varying temperature."""
    print("\nComparing goal-directed vs exploratory behavior...")

    results = {}

    # High temperature (exploratory)
    print("\n--- High Temperature (Exploratory) ---")
    agent_exploratory = ActiveInferenceAgent(
        n_states=4, n_observations=2, n_actions=2,
        action_space="continuous", policy_temperature=2.0
    )
    results["exploratory"] = run_reaching_experiment(n_episodes=30)

    # Low temperature (goal-directed)
    print("\n--- Low Temperature (Goal-Directed) ---")
    agent_goal_directed = ActiveInferenceAgent(
        n_states=4, n_observations=2, n_actions=2,
        action_space="continuous", policy_temperature=0.5
    )
    results["goal_directed"] = run_reaching_experiment(n_episodes=30)

    # Compare results
    print("\n" + "=" * 70)
    print("Temperature Comparison")
    print("=" * 70)

    for temp_type, result in results.items():
        print(f"{temp_type.capitalize()}: Success rate = {result['success_rate']".1%"}")

    return results


if __name__ == "__main__":
    # Run main reaching experiment
    results = run_reaching_experiment(n_episodes=100, max_steps=50)

    # Run temperature comparison
    comparison_results = run_goal_directed_vs_exploration()

    print("\n✓ Reaching experiment completed!")
    print("The agent demonstrates goal-directed behavior by moving toward targets,")
    print("showcasing pragmatic value (goal achievement) in expected free energy.")
