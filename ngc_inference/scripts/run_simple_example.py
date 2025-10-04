#!/usr/bin/env python3
"""
Simple Active Inference example.

Demonstrates basic free energy minimization with a simple predictive agent.
"""

import jax.numpy as jnp
from jax import random
from pathlib import Path

from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner
from ngc_inference.utils.visualization import plot_free_energy, plot_beliefs
from ngc_inference.utils.logging_config import setup_logging


def generate_sinusoidal_data(key, n_samples, n_features, frequency=0.1):
    """Generate sinusoidal data for testing."""
    t = jnp.linspace(0, n_samples * frequency, n_samples)
    data = jnp.sin(2 * jnp.pi * t[:, None] * jnp.arange(1, n_features + 1))
    
    # Add some noise
    noise_key, key = random.split(key)
    noise = 0.1 * random.normal(noise_key, data.shape)
    
    return data + noise


def main():
    """Run simple Active Inference example."""
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/simple_example.log")
    
    print("=" * 70)
    print("Simple Active Inference Example")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        "experiment": "simple_prediction_demo",
        "agent": {
            "n_observations": 10,
            "n_hidden": 15,
            "learning_rate": 0.01,
            "precision": 1.0,
        },
        "training": {
            "n_epochs": 50,
            "n_inference_steps": 20
        }
    }
    
    # Initialize
    key = random.PRNGKey(42)
    
    # Generate data
    print("Generating sinusoidal training data...")
    data = generate_sinusoidal_data(
        key,
        n_samples=50,
        n_features=config["agent"]["n_observations"]
    )
    print(f"  Data shape: {data.shape}")
    print()
    
    # Create agent
    print("Creating SimplePredictionAgent...")
    agent = SimplePredictionAgent(
        n_observations=config["agent"]["n_observations"],
        n_hidden=config["agent"]["n_hidden"],
        learning_rate=config["agent"]["learning_rate"],
        precision=config["agent"]["precision"],
        seed=42
    )
    print(f"  Observations: {agent.n_observations}")
    print(f"  Hidden units: {agent.n_hidden}")
    print()
    
    # Create simulation runner
    output_dir = "logs/runs/simple_example"
    runner = SimulationRunner(config, output_dir=output_dir)
    print(f"Output directory: {output_dir}")
    print()
    
    # Run learning
    print("Training agent...")
    print("-" * 70)
    training_results = runner.run_learning(
        agent,
        data,
        n_epochs=config["training"]["n_epochs"],
        n_inference_steps=config["training"]["n_inference_steps"],
        save_results=True
    )
    print("-" * 70)
    print()
    
    # Test inference on new data
    print("Testing inference on new observation...")
    test_key, key = random.split(key)
    test_obs = generate_sinusoidal_data(test_key, n_samples=1, n_features=10)[0:1]
    
    inference_results = runner.run_inference(
        agent,
        test_obs,
        n_steps=30,
        save_results=True
    )
    
    print(f"  Final free energy: {inference_results['metrics']['free_energy']:.4f}")
    print(f"  Prediction error: {inference_results['metrics'].get('prediction_error', inference_results['metrics'].get('final_prediction_error', 0.0)):.4f}")
    print()
    
    # Summary
    print("=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Initial loss: {training_results['losses'][0]:.4f}")
    print(f"Final loss: {training_results['losses'][-1]:.4f}")
    print(f"Improvement: {training_results['losses'][0] - training_results['losses'][-1]:.4f}")
    print()
    
    print(f"Results saved to: {output_dir}")
    print("  - config.yaml")
    print("  - learning_metrics.json")
    print("  - inference_metrics.json")
    print("  - learning_curve.png")
    print("  - free_energy.png")
    print("  - beliefs.png")
    print()
    
    print("✓ Example complete!")
    print()


if __name__ == "__main__":
    main()




