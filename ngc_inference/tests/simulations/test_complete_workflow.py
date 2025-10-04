"""
End-to-end simulation tests with orchestrators.
"""

import pytest
import jax.numpy as jnp
from jax import random
from pathlib import Path
import tempfile

from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
from ngc_inference.simulations.hierarchical_inference import HierarchicalInferenceAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner
from ngc_inference.orchestrators.experiment_manager import ExperimentManager


@pytest.mark.simulation
class TestCompleteWorkflow:
    """Test complete simulation workflows."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.temp_dir = tempfile.mkdtemp()
        
    def test_simple_simulation_workflow(self):
        """Test complete workflow with SimplePredictionAgent."""
        # Configuration
        config = {
            "agent": {
                "n_observations": 10,
                "n_hidden": 5,
                "learning_rate": 0.01
            },
            "training": {
                "n_epochs": 5,
                "n_inference_steps": 10
            }
        }
        
        # Create agent
        agent = SimplePredictionAgent(
            n_observations=config["agent"]["n_observations"],
            n_hidden=config["agent"]["n_hidden"],
            learning_rate=config["agent"]["learning_rate"],
            seed=42
        )
        
        # Create runner
        runner = SimulationRunner(config, output_dir=self.temp_dir)
        
        # Generate data
        observations = random.normal(self.key, (20, config["agent"]["n_observations"]))
        
        # Run learning
        training_results = runner.run_learning(
            agent,
            observations,
            n_epochs=config["training"]["n_epochs"],
            n_inference_steps=config["training"]["n_inference_steps"],
            save_results=True
        )
        
        assert "losses" in training_results
        assert len(training_results["losses"]) == config["training"]["n_epochs"]
        
        # Check outputs exist
        output_dir = Path(self.temp_dir)
        assert (output_dir / "config.yaml").exists()
        assert (output_dir / "learning_metrics.json").exists()
        
    def test_hierarchical_simulation_workflow(self):
        """Test complete workflow with HierarchicalInferenceAgent."""
        config = {
            "agent": {
                "layer_sizes": [10, 8, 5],
                "learning_rate": 0.005
            },
            "training": {
                "n_epochs": 5,
                "n_inference_steps": 15
            }
        }
        
        agent = HierarchicalInferenceAgent(
            layer_sizes=config["agent"]["layer_sizes"],
            learning_rate=config["agent"]["learning_rate"],
            seed=42
        )
        
        runner = SimulationRunner(config, output_dir=self.temp_dir + "/hierarchical")
        
        observations = random.normal(self.key, (15, config["agent"]["layer_sizes"][0]))
        
        training_results = runner.run_learning(
            agent,
            observations,
            n_epochs=config["training"]["n_epochs"],
            n_inference_steps=config["training"]["n_inference_steps"],
            save_results=True
        )
        
        assert "losses" in training_results
        
    def test_inference_workflow(self):
        """Test inference-only workflow."""
        config = {"agent": {"n_observations": 10, "n_hidden": 5}}
        
        agent = SimplePredictionAgent(
            n_observations=config["agent"]["n_observations"],
            n_hidden=config["agent"]["n_hidden"],
            seed=42
        )
        
        runner = SimulationRunner(config, output_dir=self.temp_dir + "/inference")
        
        observation = random.normal(self.key, (1, config["agent"]["n_observations"]))
        
        results = runner.run_inference(
            agent,
            observation,
            n_steps=20,
            save_results=True
        )
        
        assert "beliefs" in results
        assert "metrics" in results
        
    def test_experiment_manager(self):
        """Test experiment manager with parameter sweep."""
        base_config = {
            "agent": {
                "n_observations": 10,
                "n_hidden": 5
            }
        }
        
        manager = ExperimentManager(
            base_config,
            experiment_name="test_sweep",
            output_dir=self.temp_dir + "/experiments"
        )
        
        # Create parameter grid
        param_grid = {
            "agent.learning_rate": [0.01, 0.05],
            "agent.precision": [0.5, 1.0]
        }
        
        configs = manager.create_parameter_grid(param_grid)
        
        assert len(configs) == 4  # 2 x 2 combinations
        
        # Define simple run function
        def run_experiment(config, run_id):
            agent = SimplePredictionAgent(
                n_observations=config["agent"]["n_observations"],
                n_hidden=config["agent"]["n_hidden"],
                learning_rate=config["agent"]["learning_rate"],
                precision=config["agent"]["precision"],
                seed=42
            )
            
            obs = random.normal(self.key, (10, config["agent"]["n_observations"]))
            beliefs, metrics = agent.infer(obs, n_steps=5)
            
            return {
                "run_id": run_id,
                "final_loss": metrics["free_energy"],
                "config": config
            }
        
        # Run experiment
        results = manager.run_experiment(configs, run_experiment, parallel=False)
        
        assert len(results) == 4
        
        # Analyze results
        analysis = manager.analyze_results(results, metric_key="final_loss")
        assert "mean" in analysis
        assert "std" in analysis
        
        # Get best run
        best = manager.get_best_run(results, metric_key="final_loss", minimize=True)
        assert "final_loss" in best




