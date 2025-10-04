"""
Thin orchestrator for running Active Inference simulations.

Manages the execution flow of simulations with minimal overhead.
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import yaml
import jax.numpy as jnp
from datetime import datetime

from ngc_inference.utils.logging_config import get_logger
from ngc_inference.utils.visualization import plot_free_energy, plot_beliefs
from ngc_inference.utils.metrics import compute_metrics

logger = get_logger(__name__)


class SimulationRunner:
    """
    Orchestrates simulation execution with configuration management and result logging.
    
    Design principle: Thin orchestration layer that delegates to specialized components.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """
        Initialize simulation runner.
        
        Args:
            config: Simulation configuration dictionary
            output_dir: Directory for outputs (default: logs/runs/{timestamp})
        """
        self.config = config
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"logs/runs/{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        logger.info(f"SimulationRunner initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        
    def _save_config(self):
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
        
    def run_inference(
        self,
        agent: Any,
        observations: jnp.ndarray,
        n_steps: int,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference with an agent on observations.
        
        Args:
            agent: Active Inference agent
            observations: Observed data
            n_steps: Number of inference steps
            save_results: Whether to save results
            
        Returns:
            Results dictionary
        """
        logger.info(f"Running inference for {n_steps} steps")
        
        # Run inference
        beliefs, metrics = agent.infer(observations, n_steps)
        
        # Compute additional metrics
        if hasattr(agent, 'predict'):
            predictions = agent.predict(beliefs)
            extra_metrics = compute_metrics(
                observations, predictions, beliefs,
                jnp.zeros_like(beliefs)
            )
            metrics.update(extra_metrics)
        
        # Save results
        if save_results:
            self._save_results(metrics, beliefs, "inference")
            
            # Generate plots
            if "free_energy_trajectory" in metrics:
                plot_free_energy(
                    metrics["free_energy_trajectory"],
                    save_path=str(self.output_dir / "free_energy.png"),
                    show=False
                )
            
            if isinstance(beliefs, jnp.ndarray):
                plot_beliefs(
                    beliefs,
                    observations,
                    predictions if hasattr(agent, 'predict') else None,
                    save_path=str(self.output_dir / "beliefs.png"),
                    show=False
                )
        
        logger.info(f"Inference complete: FE={metrics.get('free_energy', 'N/A'):.4f}")
        
        return {"beliefs": beliefs, "metrics": metrics}
    
    def run_learning(
        self,
        agent: Any,
        observations: jnp.ndarray,
        n_epochs: int,
        n_inference_steps: int,
        save_results: bool = True,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run learning with an agent.
        
        Args:
            agent: Active Inference agent
            observations: Training data
            n_epochs: Number of training epochs
            n_inference_steps: Inference steps per sample
            save_results: Whether to save results
            callback: Optional callback function called after each epoch
            
        Returns:
            Training results
        """
        logger.info(f"Running learning for {n_epochs} epochs")
        
        # Run learning
        training_metrics = agent.learn(
            observations,
            n_epochs=n_epochs,
            n_inference_steps=n_inference_steps,
            verbose=True
        )
        
        # Call callback if provided
        if callback is not None:
            callback(training_metrics)
        
        # Save results
        if save_results:
            self._save_results(training_metrics, None, "learning")
            
            # Plot learning curve
            if "losses" in training_metrics:
                plot_free_energy(
                    training_metrics["losses"],
                    title="Learning Curve (Training Loss)",
                    save_path=str(self.output_dir / "learning_curve.png"),
                    show=False
                )
        
        logger.info(f"Learning complete: Final loss={training_metrics.get('final_loss', 'N/A'):.4f}")
        
        return training_metrics
    
    def _save_results(
        self,
        metrics: Dict,
        beliefs: Optional[jnp.ndarray],
        prefix: str
    ):
        """Save results to JSON and numpy files."""
        # Save metrics (converting arrays to lists for JSON)
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                metrics_json[key] = value
            elif isinstance(value, (int, float)):
                metrics_json[key] = value
            elif isinstance(value, jnp.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                metrics_json[key] = {
                    k: v.tolist() if isinstance(v, jnp.ndarray) else v
                    for k, v in value.items()
                }
        
        metrics_path = self.output_dir / f"{prefix}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        logger.debug(f"Metrics saved to {metrics_path}")
        
        # Save beliefs if provided
        if beliefs is not None:
            beliefs_path = self.output_dir / f"{prefix}_beliefs.npy"
            jnp.save(beliefs_path, beliefs)
            logger.debug(f"Beliefs saved to {beliefs_path}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all results in output directory."""
        summary = {
            "output_dir": str(self.output_dir),
            "config": self.config,
            "files": list(self.output_dir.glob("*"))
        }
        
        # Load metrics if available
        metrics_files = list(self.output_dir.glob("*_metrics.json"))
        if metrics_files:
            summary["metrics"] = {}
            for mf in metrics_files:
                with open(mf) as f:
                    summary["metrics"][mf.stem] = json.load(f)
        
        return summary




