"""
Simple Active Inference simulation: single-layer predictive model.

Demonstrates basic variational free energy minimization for learning to predict
sensory observations from a generative model.
"""

from typing import Dict, Tuple, Optional
import jax.numpy as jnp
from jax import random, jit
from ngclearn import Context
from ngclearn.components import RateCell, GaussianErrorCell, DenseSynapse
from ngcsimlib.compilers.process import Process

from ngc_inference.core.free_energy import compute_free_energy, compute_state_gradients
from ngc_inference.utils.logging_config import get_logger
from ngc_inference.utils.metrics import InferenceMetrics

logger = get_logger(__name__)


class SimplePredictionAgent:
    """
    Simple predictive coding agent for learning sensory predictions.
    
    Architecture:
    - Input layer: sensory observations
    - Hidden layer: latent representations
    - Prediction layer: reconstructed observations
    - Error neurons: compute prediction errors
    
    Learning: Minimize variational free energy through gradient descent.
    """
    
    def __init__(
        self,
        n_observations: int,
        n_hidden: int,
        batch_size: int = 1,
        learning_rate: float = 0.01,
        precision: float = 1.0,
        tau: float = 10.0,
        dt: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize simple prediction agent.
        
        Args:
            n_observations: Observation dimensionality
            n_hidden: Hidden state dimensionality
            batch_size: Batch size
            learning_rate: Learning rate for parameters
            precision: Observation precision (inverse variance)
            tau: Time constant for neuronal dynamics
            dt: Integration time step
            seed: Random seed
        """
        self.n_observations = n_observations
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.precision = precision
        self.tau = tau
        self.dt = dt
        self.key = random.PRNGKey(seed)
        
        self.metrics = InferenceMetrics()
        
        logger.info(f"Initializing SimplePredictionAgent")
        logger.info(f"  Architecture: obs={n_observations}, hidden={n_hidden}")
        logger.info(f"  Parameters: lr={learning_rate}, precision={precision}")
        
        self._build_model()
        
    def _build_model(self):
        """Build the predictive coding network."""
        import uuid
        context_name = f"simple_prediction_{uuid.uuid4().hex[:8]}"
        with Context(context_name) as ctx:
            # Hidden state (latent representation)
            self.z_hidden = RateCell(
                name="z_hidden",
                n_units=self.n_hidden,
                batch_size=self.batch_size,
                tau_m=self.tau,
                act_fx="identity"
            )
            
            # Observation prediction
            self.z_pred = RateCell(
                name="z_pred",
                n_units=self.n_observations,
                batch_size=self.batch_size,
                tau_m=self.tau,
                act_fx="identity"
            )
            
            # Prediction error neurons
            self.e_obs = GaussianErrorCell(
                name="e_obs",
                n_units=self.n_observations,
                batch_size=self.batch_size,
                sigma=1.0 / jnp.sqrt(self.precision)
            )
            
            # Generative (top-down) weights
            self.W_gen = DenseSynapse(
                name="W_gen",
                shape=(self.n_hidden, self.n_observations),
                eta=self.learning_rate,
                weight_init={"dist": "gaussian", "mu": 0.0, "sigma": 0.1},
                bias_init=None,
                w_bound=1.0
            )
            
            # Recognition (bottom-up) weights
            self.W_rec = DenseSynapse(
                name="W_rec",
                shape=(self.n_observations, self.n_hidden),
                eta=self.learning_rate,
                weight_init={"dist": "gaussian", "mu": 0.0, "sigma": 0.1},
                bias_init=None,
                w_bound=1.0
            )
            
            # Wire the network
            # Top-down prediction: z_hidden -> W_gen -> z_pred
            self.W_gen.inputs << self.z_hidden.zF
            self.z_pred.j << self.W_gen.outputs
            
            # Prediction error
            self.e_obs.mu << self.z_pred.zF
            
            # Bottom-up inference: e_obs -> W_rec -> z_hidden
            self.W_rec.inputs << self.e_obs.dmu
            self.z_hidden.j << self.W_rec.outputs
            
            # Compile processes
            advance_process = (
                Process("advance")
                >> self.e_obs.advance_state
                >> self.W_rec.advance_state
                >> self.z_hidden.advance_state
                >> self.W_gen.advance_state
                >> self.z_pred.advance_state
            )
            ctx.wrap_and_add_command(jit(advance_process.pure), name="advance")
            
            reset_process = (
                Process("reset")
                >> self.e_obs.reset
                >> self.W_rec.reset
                >> self.z_hidden.reset
                >> self.W_gen.reset
                >> self.z_pred.reset
            )
            ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
            
            @Context.dynamicCommand
            def clamp_observation(obs):
                """Clamp observation."""
                self.e_obs.target.set(obs)
        
        self.context = ctx
        logger.info("Model built successfully")
        
    def infer(
        self,
        observation: jnp.ndarray,
        n_steps: int = 20
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Perform inference on a single observation using proper VFE minimization.

        Uses gradient descent on beliefs to minimize variational free energy.

        Args:
            observation: Observed data (batch_size, n_observations)
            n_steps: Number of inference iterations

        Returns:
            beliefs: Inferred hidden states
            metrics: Inference metrics
        """
        self.context.reset()

        # Initialize beliefs
        beliefs = self.z_hidden.zF.value.copy()
        fe_trajectory = []

        for step in range(n_steps):
            # Clamp observation
            self.context.clamp_observation(observation)

            # Get current state and prediction
            pred = self.z_pred.zF.value
            prior = jnp.zeros_like(beliefs)

            # Compute free energy and gradients
            fe, fe_components = compute_free_energy(
                observation, pred, prior, beliefs,
                self.precision, 1.0
            )

            # Compute gradients for belief updates
            grad = compute_state_gradients(
                observation, beliefs, self.W_gen.weights.value, prior,
                self.precision, 1.0
            )

            # Gradient descent update on beliefs
            beliefs = beliefs - self.learning_rate * self.dt * grad

            # Update hidden state in the model
            self.z_hidden.z.set(beliefs)

            # Advance the model to update predictions
            self.context.advance(t=step * self.dt, dt=self.dt)

            fe_trajectory.append(float(fe))

        # Final state after inference
        final_beliefs = self.z_hidden.zF.value
        final_prediction = self.z_pred.zF.value

        metrics = {
            "free_energy": fe_trajectory[-1],
            "free_energy_trajectory": fe_trajectory,
            "prediction_error": float(jnp.mean(jnp.square(observation - final_prediction))),
            "beliefs": final_beliefs,
            "prediction": final_prediction
        }

        return final_beliefs, metrics
    
    def learn(
        self,
        observations: jnp.ndarray,
        n_epochs: int = 100,
        n_inference_steps: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Learn generative model from observations.
        
        Args:
            observations: Training data (n_samples, n_observations)
            n_epochs: Number of training epochs
            n_inference_steps: Inference steps per sample
            verbose: Whether to print progress
            
        Returns:
            Training metrics
        """
        self.metrics.reset()
        
        n_samples = observations.shape[0]
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for i in range(n_samples):
                obs = observations[i:i+1]  # Keep batch dimension
                
                beliefs, infer_metrics = self.infer(obs, n_inference_steps)
                epoch_loss += infer_metrics["free_energy"]
            
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss:.4f}")
        
        return {
            "losses": losses,
            "final_loss": losses[-1],
            "weights_gen": self.W_gen.weights.value,
            "weights_rec": self.W_rec.weights.value
        }
    
    def predict(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """
        Generate prediction from hidden states.
        
        Args:
            hidden_states: Hidden state values
            
        Returns:
            Predicted observations
        """
        return jnp.dot(hidden_states, self.W_gen.weights.value)




