"""
Experiment manager for running multiple simulations with parameter sweeps.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from ngc_inference.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExperimentManager:
    """
    Manages multiple simulation runs with parameter sweeps and experiment tracking.
    
    Supports:
    - Parameter grid search
    - Parallel execution
    - Result aggregation
    - Experiment comparison
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        experiment_name: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize experiment manager.
        
        Args:
            base_config: Base configuration shared across all runs
            experiment_name: Name of the experiment
            output_dir: Root directory for experiments
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"logs/experiments/{experiment_name}_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs = []
        
        logger.info(f"ExperimentManager initialized: {experiment_name}")
        logger.info(f"  Output directory: {self.output_dir}")
        
    def create_parameter_grid(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create list of configurations from parameter grid.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of configuration dictionaries
        """
        # Get all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        configs = []
        for combination in itertools.product(*values):
            config = self.base_config.copy()
            for key, value in zip(keys, combination):
                # Support nested keys with dot notation (e.g., "agent.learning_rate")
                keys_nested = key.split('.')
                target = config
                for k in keys_nested[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys_nested[-1]] = value
            configs.append(config)
        
        logger.info(f"Created parameter grid: {len(configs)} configurations")
        return configs
    
    def run_experiment(
        self,
        configs: List[Dict[str, Any]],
        run_function: callable,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run experiment with multiple configurations.
        
        Args:
            configs: List of configuration dictionaries
            run_function: Function that takes config and run_id, returns results
            parallel: Whether to run in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of results for each run
        """
        logger.info(f"Starting experiment with {len(configs)} runs")
        
        results = []
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for run_id, config in enumerate(configs):
                    future = executor.submit(run_function, config, run_id)
                    futures[future] = run_id
                
                for future in as_completed(futures):
                    run_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Run {run_id} completed successfully")
                    except Exception as e:
                        logger.error(f"Run {run_id} failed: {e}")
                        results.append({"run_id": run_id, "error": str(e)})
        else:
            for run_id, config in enumerate(configs):
                try:
                    result = run_function(config, run_id)
                    results.append(result)
                    logger.info(f"Run {run_id}/{len(configs)} completed")
                except Exception as e:
                    logger.error(f"Run {run_id} failed: {e}")
                    results.append({"run_id": run_id, "error": str(e)})
        
        # Save experiment results
        self._save_experiment_results(results)
        
        logger.info(f"Experiment complete: {len(results)} runs finished")
        return results
    
    def _save_experiment_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to file."""
        results_path = self.output_dir / "experiment_results.json"
        
        # Convert arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):
                    serializable[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable[key] = {
                        k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in value.items()
                    }
                else:
                    serializable[key] = value
            serializable_results.append(serializable)
        
        with open(results_path, 'w') as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "n_runs": len(results),
                "results": serializable_results
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def analyze_results(
        self,
        results: List[Dict[str, Any]],
        metric_key: str = "final_loss"
    ) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            results: List of result dictionaries
            metric_key: Key of metric to analyze
            
        Returns:
            Analysis summary
        """
        import jax.numpy as jnp
        
        # Extract metric values
        values = []
        for result in results:
            if metric_key in result and "error" not in result:
                values.append(result[metric_key])
        
        if not values:
            logger.warning(f"No valid values found for metric: {metric_key}")
            return {}
        
        values = jnp.array(values)
        
        analysis = {
            "metric": metric_key,
            "n_runs": len(values),
            "mean": float(jnp.mean(values)),
            "std": float(jnp.std(values)),
            "min": float(jnp.min(values)),
            "max": float(jnp.max(values)),
            "median": float(jnp.median(values)),
        }
        
        # Save analysis
        analysis_path = self.output_dir / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis: {metric_key} = {analysis['mean']:.4f} ± {analysis['std']:.4f}")
        
        return analysis
    
    def get_best_run(
        self,
        results: List[Dict[str, Any]],
        metric_key: str = "final_loss",
        minimize: bool = True
    ) -> Dict[str, Any]:
        """
        Get the best performing run.
        
        Args:
            results: List of result dictionaries
            metric_key: Key of metric to optimize
            minimize: Whether to minimize (True) or maximize (False)
            
        Returns:
            Best result dictionary
        """
        valid_results = [r for r in results if metric_key in r and "error" not in r]
        
        if not valid_results:
            logger.warning("No valid results found")
            return {}
        
        if minimize:
            best = min(valid_results, key=lambda r: r[metric_key])
        else:
            best = max(valid_results, key=lambda r: r[metric_key])
        
        logger.info(f"Best run: {metric_key} = {best[metric_key]:.4f}")
        
        return best






