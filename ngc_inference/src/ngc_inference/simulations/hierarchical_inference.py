"""
Hierarchical Active Inference simulation: multi-layer predictive hierarchy.

Implements a hierarchical generative model where each layer predicts the layer below,
demonstrating deep Active Inference with variational free energy minimization.
"""

from typing import Dict, Tuple, List, Optional
import jax.numpy as jnp
from jax import random, jit
from ngclearn import Context
from ngclearn.components import RateCell, GaussianErrorCell, DenseSynapse
from ngcsimlib.compilers.process import Process

from ngc_inference.core.free_energy import compute_free_energy, compute_state_gradients
from ngc_inference.utils.logging_config import get_logger
from ngc_inference.utils.metrics import InferenceMetrics

logger = get_logger(__name__)


class HierarchicalInferenceAgent:
    """
    Hierarchical predictive coding agent with multiple layers of representation.
    
    Architecture:
    - Multiple layers of hidden states (z1, z2, ..., zN)
    - Each layer predicts the layer below
    - Error neurons at each level compute prediction errors
    - Bottom layer predicts observations
    
    Learning: Hierarchical free energy minimization through predictive coding.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        batch_size: int = 1,
        learning_rate: float = 0.01,
        precisions: Optional[List[float]] = None,
        tau: float = 10.0,
        dt: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize hierarchical inference agent.
        
        Args:
            layer_sizes: List of layer dimensions [obs, hidden1, hidden2, ...]
            batch_size: Batch size
            learning_rate: Learning rate
            precisions: Precision at each level (if None, use 1.0 for all)
            tau: Time constant
            dt: Integration time step
            seed: Random seed
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # Number of hidden layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.dt = dt
        self.key = random.PRNGKey(seed)
        
        if precisions is None:
            self.precisions = [1.0] * len(layer_sizes)
        else:
            self.precisions = precisions
        
        self.metrics = InferenceMetrics()
        
        logger.info(f"Initializing HierarchicalInferenceAgent")
        logger.info(f"  Architecture: {layer_sizes}")
        logger.info(f"  Layers: {self.n_layers}, LR: {learning_rate}")
        
        self._build_model()
        
    def _build_model(self):
        """Build hierarchical predictive coding network."""
        import uuid
        context_name = f"hierarchical_inference_{uuid.uuid4().hex[:8]}"
        with Context(context_name) as ctx:
            self.states = []
            self.predictions = []
            self.errors = []
            self.gen_synapses = []
            self.rec_synapses = []
            
            # Build layers bottom-up
            for layer_idx in range(self.n_layers + 1):
                n_units = self.layer_sizes[layer_idx]
                
                # State neurons
                if layer_idx > 0:  # Hidden layers
                    state = RateCell(
                        name=f"z{layer_idx}",
                        n_units=n_units,
                        batch_size=self.batch_size,
                        tau_m=self.tau,
                        act_fx="tanh"
                    )
                    self.states.append(state)
                
                # Prediction neurons (for layers 0 to n_layers-1)
                if layer_idx < self.n_layers:
                    pred = RateCell(
                        name=f"pred{layer_idx}",
                        n_units=n_units,
                        batch_size=self.batch_size,
                        tau_m=self.tau,
                        act_fx="identity"
                    )
                    self.predictions.append(pred)
                    
                    # Error neurons
                    error = GaussianErrorCell(
                        name=f"error{layer_idx}",
                        n_units=n_units,
                        batch_size=self.batch_size,
                        sigma=1.0 / jnp.sqrt(self.precisions[layer_idx])
                    )
                    self.errors.append(error)
            
            # Build connections (top-down and bottom-up)
            for layer_idx in range(self.n_layers):
                n_upper = self.layer_sizes[layer_idx + 1]
                n_lower = self.layer_sizes[layer_idx]
                
                # Top-down generative connections
                W_gen = DenseSynapse(
                    name=f"W_gen_{layer_idx}",
                    shape=(n_upper, n_lower),
                    eta=self.learning_rate,
                    weight_init={"dist": "gaussian", "mu": 0.0, "sigma": 0.1},
                    bias_init=None,
                    w_bound=1.0
                )
                self.gen_synapses.append(W_gen)
                
                # Bottom-up recognition connections
                W_rec = DenseSynapse(
                    name=f"W_rec_{layer_idx}",
                    shape=(n_lower, n_upper),
                    eta=self.learning_rate,
                    weight_init={"dist": "gaussian", "mu": 0.0, "sigma": 0.1},
                    bias_init=None,
                    w_bound=1.0
                )
                self.rec_synapses.append(W_rec)
                
                # Wire generative pathway: state -> W_gen -> prediction
                W_gen.inputs << self.states[layer_idx].zF
                self.predictions[layer_idx].j << W_gen.outputs
                
                # Wire error computation
                self.errors[layer_idx].mu << self.predictions[layer_idx].zF
                
                # Wire recognition pathway: error -> W_rec -> state
                W_rec.inputs << self.errors[layer_idx].dmu
                if layer_idx < self.n_layers - 1:
                    self.states[layer_idx].j << W_rec.outputs
                else:
                    # Top layer receives bottom-up input
                    self.states[layer_idx].j << W_rec.outputs
            
            # Compile processes
            advance_components = []
            reset_components = []
            
            # Add components in proper order for computation
            for error in self.errors:
                advance_components.append(error.advance_state)
                reset_components.append(error.reset)
            
            for synapse in self.rec_synapses:
                advance_components.append(synapse.advance_state)
                reset_components.append(synapse.reset)
            
            for state in self.states:
                advance_components.append(state.advance_state)
                reset_components.append(state.reset)
            
            for synapse in self.gen_synapses:
                advance_components.append(synapse.advance_state)
                reset_components.append(synapse.reset)
            
            for pred in self.predictions:
                advance_components.append(pred.advance_state)
                reset_components.append(pred.reset)
            
            # Build processes
            advance_process = Process("advance")
            for component in advance_components:
                advance_process = advance_process >> component
            ctx.wrap_and_add_command(jit(advance_process.pure), name="advance")
            
            reset_process = Process("reset")
            for component in reset_components:
                reset_process = reset_process >> component
            ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
            
            @Context.dynamicCommand
            def clamp_observation(obs):
                """Clamp observation at lowest level."""
                self.errors[0].target.set(obs)
        
        self.context = ctx
        logger.info("Hierarchical model built successfully")
        
    def infer(
        self,
        observation: jnp.ndarray,
        n_steps: int = 30
    ) -> Tuple[List[jnp.ndarray], Dict]:
        """
        Perform hierarchical inference using proper VFE minimization.

        Uses gradient descent on beliefs at each level to minimize hierarchical free energy.

        Args:
            observation: Observed data
            n_steps: Number of inference iterations

        Returns:
            beliefs: List of inferred states at each level
            metrics: Inference metrics
        """
        self.context.reset()

        # Initialize beliefs for all layers
        beliefs = [state.zF.value.copy() for state in self.states]
        fe_trajectory = []

        for step in range(n_steps):
            # Clamp observation
            self.context.clamp_observation(observation)

            # Compute hierarchical free energy and update each layer
            total_fe = 0.0

            for layer_idx in range(self.n_layers):
                if layer_idx == 0:
                    target = observation
                else:
                    target = beliefs[layer_idx - 1]

                # Get current state and prediction
                state = beliefs[layer_idx]
                pred = self.predictions[layer_idx].zF.value
                prior = jnp.zeros_like(state)

                # Compute free energy for this layer
                fe, _ = compute_free_energy(
                    target, pred, prior, state,
                    self.precisions[layer_idx],
                    self.precisions[layer_idx + 1]
                )
                total_fe += float(fe)

                # Compute gradients for this layer's beliefs
                grad = compute_state_gradients(
                    target, state, self.gen_synapses[layer_idx].weights.value, prior,
                    self.precisions[layer_idx], self.precisions[layer_idx + 1]
                )

                # Gradient descent update on this layer's beliefs
                beliefs[layer_idx] = beliefs[layer_idx] - self.learning_rate * self.dt * grad

                # Update the state in the model
                self.states[layer_idx].z.set(beliefs[layer_idx])

            # Advance the model to update predictions
            self.context.advance(t=step * self.dt, dt=self.dt)

            fe_trajectory.append(total_fe)

        # Collect final beliefs from all layers
        final_beliefs = [state.zF.value for state in self.states]
        final_predictions = [pred.zF.value for pred in self.predictions]

        metrics = {
            "free_energy": fe_trajectory[-1],
            "free_energy_trajectory": fe_trajectory,
            "beliefs": final_beliefs,
            "predictions": final_predictions,
            "prediction_error_L0": float(jnp.mean(jnp.square(self.errors[0].dmu.value)))
        }

        return final_beliefs, metrics
    
    def learn(
        self,
        observations: jnp.ndarray,
        n_epochs: int = 100,
        n_inference_steps: int = 30,
        verbose: bool = True
    ) -> Dict:
        """
        Learn hierarchical generative model.
        
        Args:
            observations: Training data
            n_epochs: Number of epochs
            n_inference_steps: Inference steps per sample
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        self.metrics.reset()
        
        n_samples = observations.shape[0]
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for i in range(n_samples):
                obs = observations[i:i+1]
                
                beliefs, infer_metrics = self.infer(obs, n_inference_steps)
                epoch_loss += infer_metrics["free_energy"]
            
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss:.4f}")
        
        # Collect learned weights
        weights = {
            f"gen_layer_{i}": syn.weights.value 
            for i, syn in enumerate(self.gen_synapses)
        }
        weights.update({
            f"rec_layer_{i}": syn.weights.value 
            for i, syn in enumerate(self.rec_synapses)
        })
        
        return {
            "losses": losses,
            "final_loss": losses[-1],
            "weights": weights
        }
    
    def generate(self, top_level_state: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Generate observations from top-level state (top-down generation).
        
        Args:
            top_level_state: State at highest level
            
        Returns:
            Generated predictions at all levels
        """
        predictions = []
        current_state = top_level_state
        
        for layer_idx in range(self.n_layers - 1, -1, -1):
            pred = jnp.dot(current_state, self.gen_synapses[layer_idx].weights.value)
            predictions.insert(0, pred)
            current_state = pred
        
        return predictions




