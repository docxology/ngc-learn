"""
Active Inference agent implementations using ngclearn components.

Provides base classes for variational and active inference agents that minimize
free energy through predictive coding.
"""

from typing import Dict, Tuple, Optional, Any
import jax.numpy as jnp
from jax import random, jit
from ngclearn import Context
from ngclearn.components import RateCell, GaussianErrorCell, DenseSynapse
from ngcsimlib.compilers.process import Process

from ngc_inference.core.free_energy import (
    compute_free_energy,
    compute_prediction_error,
    compute_expected_free_energy
)
from ngc_inference.utils.logging_config import get_logger

logger = get_logger(__name__)


class VariationalInferenceAgent:
    """
    Base class for variational inference using predictive coding.
    
    Implements hierarchical inference by minimizing prediction errors across layers,
    following the Free Energy Principle.
    """
    
    def __init__(
        self,
        name: str,
        n_observations: int,
        n_hidden: int,
        batch_size: int = 1,
        learning_rate: float = 0.01,
        precision_obs: float = 1.0,
        precision_hidden: float = 1.0,
        integration_steps: int = 10,
        dt: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize variational inference agent.
        
        Args:
            name: Unique identifier for the agent
            n_observations: Dimensionality of observations
            n_hidden: Dimensionality of hidden states
            batch_size: Batch size for processing
            learning_rate: Learning rate for weight updates
            precision_obs: Observation precision (inverse variance)
            precision_hidden: Hidden state precision
            integration_steps: Number of inference iterations
            dt: Integration time step
            seed: Random seed
        """
        self.name = name
        self.n_observations = n_observations
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.precision_obs = precision_obs
        self.precision_hidden = precision_hidden
        self.integration_steps = integration_steps
        self.dt = dt
        self.key = random.PRNGKey(seed)
        
        logger.info(f"Initializing {self.__class__.__name__}: {name}")
        logger.info(f"  n_observations={n_observations}, n_hidden={n_hidden}")
        
        self._build_model()
        
    def _build_model(self):
        """Build the inference model using ngclearn components."""
        with Context(self.name) as ctx:
            # Hidden state neurons (beliefs)
            self.z_hidden = RateCell(
                name="z_hidden",
                n_units=self.n_hidden,
                batch_size=self.batch_size,
                tau_m=self.dt,
                prior=jnp.zeros((1, self.n_hidden))
            )
            
            # Observation prediction neurons
            self.z_obs_pred = RateCell(
                name="z_obs_pred",
                n_units=self.n_observations,
                batch_size=self.batch_size,
                tau_m=self.dt
            )
            
            # Error neurons for observations
            self.e_obs = GaussianErrorCell(
                name="e_obs",
                n_units=self.n_observations,
                batch_size=self.batch_size,
                sigma=1.0 / jnp.sqrt(self.precision_obs)
            )
            
            # Generative weights (hidden -> observation prediction)
            self.W_gen = DenseSynapse(
                name="W_gen",
                shape=(self.n_hidden, self.n_observations),
                eta=self.learning_rate,
                weight_init={"dist": "gaussian", "mu": 0.0, "sigma": 0.1},
                bias_init=None,
                w_bound=1.0
            )
            
            # Wire the network
            # Forward prediction: z_hidden -> W_gen -> z_obs_pred
            self.W_gen.inputs << self.z_hidden.zF
            self.z_obs_pred.j << self.W_gen.outputs
            
            # Prediction error computation
            self.e_obs.mu << self.z_obs_pred.zF
            
            # Compile processes
            advance_process = (
                Process("advance") 
                >> self.z_hidden.advance_state
                >> self.W_gen.advance_state
                >> self.z_obs_pred.advance_state
                >> self.e_obs.advance_state
            )
            ctx.wrap_and_add_command(jit(advance_process.pure), name="advance")
            
            reset_process = (
                Process("reset")
                >> self.z_hidden.reset
                >> self.z_obs_pred.reset
                >> self.e_obs.reset
                >> self.W_gen.reset
            )
            ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
            
            @Context.dynamicCommand
            def clamp_observation(obs):
                """Clamp observation to error neuron."""
                self.e_obs.target.set(obs)
                
            @Context.dynamicCommand
            def update_hidden(error_signal):
                """Update hidden state based on prediction error."""
                current = self.z_hidden.zF.value
                delta = -error_signal * self.learning_rate * self.dt
                self.z_hidden.z.set(current + delta)
        
        self.context = ctx
        logger.info(f"Model built successfully for {self.name}")
        
    def infer(
        self,
        observations: jnp.ndarray,
        n_steps: Optional[int] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Perform inference on observations.
        
        Args:
            observations: Observed data (batch_size, n_observations)
            n_steps: Number of inference iterations (default: self.integration_steps)
            
        Returns:
            beliefs: Inferred hidden states
            metrics: Dictionary of inference metrics
        """
        if n_steps is None:
            n_steps = self.integration_steps
            
        self.context.reset()
        
        free_energies = []
        
        for step in range(n_steps):
            # Clamp observations
            self.context.clamp_observation(observations)
            
            # Advance network
            self.context.advance(t=step * self.dt, dt=self.dt)
            
            # Compute free energy
            prediction = self.z_obs_pred.zF.value
            hidden_state = self.z_hidden.zF.value
            prior_mean = jnp.zeros_like(hidden_state)
            
            fe, fe_components = compute_free_energy(
                observations,
                prediction,
                prior_mean,
                hidden_state,
                self.precision_obs,
                self.precision_hidden
            )
            free_energies.append(float(fe))
            
            # Update hidden states based on error
            error = self.e_obs.dmu.value
            self.context.update_hidden(error)
        
        beliefs = self.z_hidden.zF.value
        
        metrics = {
            "free_energy": free_energies[-1],
            "free_energy_trajectory": free_energies,
            "final_prediction_error": float(jnp.mean(jnp.square(self.e_obs.dmu.value))),
            "beliefs": beliefs
        }
        
        logger.debug(f"Inference complete: FE={free_energies[-1]:.4f}")
        
        return beliefs, metrics
    
    def learn(
        self,
        observations: jnp.ndarray,
        n_epochs: int = 1
    ) -> Dict[str, list]:
        """
        Learn generative model parameters from observations.
        
        Args:
            observations: Training data
            n_epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        metrics = {"loss": [], "free_energy": []}
        
        for epoch in range(n_epochs):
            beliefs, infer_metrics = self.infer(observations)
            
            # Hebbian-like weight update
            # This would typically be handled by ngclearn's learning rules
            
            metrics["loss"].append(infer_metrics["final_prediction_error"])
            metrics["free_energy"].append(infer_metrics["free_energy"])
            
            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}: FE={metrics['free_energy'][-1]:.4f}")
        
        return metrics


class ActiveInferenceAgent(VariationalInferenceAgent):
    """
    Active Inference agent that selects actions to minimize expected free energy.
    
    Extends variational inference with action selection based on expected outcomes.
    """
    
    def __init__(
        self,
        name: str,
        n_observations: int,
        n_hidden: int,
        n_actions: int,
        batch_size: int = 1,
        learning_rate: float = 0.01,
        precision_obs: float = 1.0,
        precision_hidden: float = 1.0,
        integration_steps: int = 10,
        dt: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize active inference agent.
        
        Args:
            n_actions: Number of available actions
            (other args same as VariationalInferenceAgent)
        """
        self.n_actions = n_actions
        super().__init__(
            name, n_observations, n_hidden, batch_size,
            learning_rate, precision_obs, precision_hidden,
            integration_steps, dt, seed
        )
        
    def select_action(
        self,
        current_observation: jnp.ndarray,
        preferred_observation: jnp.ndarray,
        available_actions: Optional[jnp.ndarray] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action that minimizes expected free energy.
        
        Args:
            current_observation: Current observation
            preferred_observation: Goal/desired observation
            available_actions: Available action indices
            
        Returns:
            selected_action: Index of selected action
            metrics: Action selection metrics
        """
        if available_actions is None:
            available_actions = jnp.arange(self.n_actions)
        
        # Infer current state
        beliefs, _ = self.infer(current_observation)
        
        expected_free_energies = []
        
        # Evaluate each action
        for action in available_actions:
            # Predict outcome of action (simplified)
            # In full implementation, would use learned transition model
            predicted_obs = self._predict_observation(beliefs, action)
            
            # Compute expected free energy
            efe, efe_components = compute_expected_free_energy(
                predicted_obs,
                preferred_observation,
                observation_precision=self.precision_obs
            )
            expected_free_energies.append(float(efe))
        
        # Select action with minimum expected free energy
        selected_idx = jnp.argmin(jnp.array(expected_free_energies))
        selected_action = int(available_actions[selected_idx])
        
        metrics = {
            "selected_action": selected_action,
            "expected_free_energies": expected_free_energies,
            "beliefs": beliefs
        }
        
        logger.debug(f"Action selected: {selected_action} (EFE={expected_free_energies[selected_idx]:.4f})")
        
        return selected_action, metrics
    
    def _predict_observation(
        self,
        state: jnp.ndarray,
        action: int
    ) -> jnp.ndarray:
        """
        Predict observation given state and action.
        
        Args:
            state: Current state beliefs
            action: Action to take
            
        Returns:
            Predicted observation
        """
        # Simplified prediction using generative weights
        # In full implementation, would include action-dependent dynamics
        prediction = jnp.dot(state, self.W_gen.weights.value)
        return prediction


