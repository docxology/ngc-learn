#!/usr/bin/env python3
"""
Hierarchical Active Inference example.

Demonstrates multi-level predictive coding with hierarchical free energy minimization.
"""

import jax.numpy as jnp
from jax import random
from pathlib import Path

from ngc_inference.simulations.hierarchical_inference import HierarchicalInferenceAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner
from ngc_inference.utils.logging_config import setup_logging


def generate_hierarchical_data(key, n_samples, n_features, hierarchy_levels=3):
    """
    Generate hierarchically structured data.
    
    Creates data with multiple levels of abstraction:
    - Low level: Fast oscillations
    - Mid level: Medium oscillations
    - High level: Slow trends
    """
    t = jnp.linspace(0, 10, n_samples)
    
    # High-level slow component
    high_level = jnp.sin(0.5 * t)
    
    # Mid-level medium component
    mid_level = 0.5 * jnp.sin(2.0 * t)
    
    # Low-level fast component
    low_level = 0.25 * jnp.sin(8.0 * t)
    
    # Combine levels
    data = (high_level + mid_level + low_level)[:, None]
    data = jnp.tile(data, (1, n_features))
    
    # Add phase shifts across features
    for i in range(n_features):
        data = data.at[:, i].set(
            data[:, i] + 0.1 * jnp.sin(2.0 * jnp.pi * i / n_features * t)
        )
    
    # Add noise
    noise_key, key = random.split(key)
    noise = 0.05 * random.normal(noise_key, data.shape)
    
    return data + noise


def main():
    """Run hierarchical Active Inference example."""
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/hierarchical_example.log")
    
    print("=" * 70)
    print("Hierarchical Active Inference Example")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        "experiment": "hierarchical_inference_demo",
        "agent": {
            "layer_sizes": [10, 15, 12, 8],  # 3-layer hierarchy
            "learning_rate": 0.005,
            "precisions": [1.0, 1.0, 1.0, 1.0],
        },
        "training": {
            "n_epochs": 80,
            "n_inference_steps": 30
        }
    }
    
    # Initialize
    key = random.PRNGKey(42)
    
    # Generate hierarchical data
    print("Generating hierarchically structured data...")
    data = generate_hierarchical_data(
        key,
        n_samples=80,
        n_features=config["agent"]["layer_sizes"][0]
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Hierarchy: {len(config['agent']['layer_sizes']) - 1} levels")
    print()
    
    # Create hierarchical agent
    print("Creating HierarchicalInferenceAgent...")
    agent = HierarchicalInferenceAgent(
        layer_sizes=config["agent"]["layer_sizes"],
        learning_rate=config["agent"]["learning_rate"],
        precisions=config["agent"]["precisions"],
        seed=42
    )
    print(f"  Layer sizes: {agent.layer_sizes}")
    print(f"  Number of layers: {agent.n_layers}")
    print()
    
    # Create simulation runner
    output_dir = "logs/runs/hierarchical_example"
    runner = SimulationRunner(config, output_dir=output_dir)
    print(f"Output directory: {output_dir}")
    print()
    
    # Run learning
    print("Training hierarchical agent...")
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
    print("Testing hierarchical inference on new observation...")
    test_key, key = random.split(key)
    test_obs = generate_hierarchical_data(
        test_key, n_samples=1, n_features=config["agent"]["layer_sizes"][0]
    )[0:1]
    
    beliefs, inference_metrics = agent.infer(test_obs, n_steps=50)
    
    print(f"  Final free energy: {inference_metrics['free_energy']:.4f}")
    print(f"  Prediction error (L0): {inference_metrics['prediction_error_L0']:.4f}")
    print(f"  Number of belief levels: {len(beliefs)}")
    for i, belief in enumerate(beliefs):
        print(f"    Level {i+1} shape: {belief.shape}")
    print()
    
    # Test top-down generation
    print("Testing top-down generation from high-level state...")
    top_key, key = random.split(key)
    top_state = random.normal(top_key, (1, config["agent"]["layer_sizes"][-1]))
    
    generated = agent.generate(top_state)
    print(f"  Generated {len(generated)} levels of predictions")
    print(f"  Final observation prediction shape: {generated[0].shape}")
    print()
    
    # Training summary
    print("=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Initial loss: {training_results['losses'][0]:.4f}")
    print(f"Final loss: {training_results['losses'][-1]:.4f}")
    print(f"Improvement: {training_results['losses'][0] - training_results['losses'][-1]:.4f}")
    print()
    
    # Weight statistics
    print("Learned Weight Statistics:")
    for key, weights in training_results['weights'].items():
        print(f"  {key}: shape={weights.shape}, mean={jnp.mean(weights):.4f}, std={jnp.std(weights):.4f}")
    print()
    
    print(f"Results saved to: {output_dir}")
    print("  - config.yaml")
    print("  - learning_metrics.json")
    print("  - learning_curve.png")
    print()
    
    print("✓ Hierarchical example complete!")
    print()


if __name__ == "__main__":
    main()






