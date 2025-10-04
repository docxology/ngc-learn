"""
Test real ngclearn integration and functionality.
"""

import pytest


@pytest.mark.integration
class TestNgclearnIntegration:
    """Test that real ngclearn components work correctly."""
    
    def test_ngclearn_import(self):
        """Test that ngclearn imports successfully."""
        try:
            import ngclearn
            import ngcsimlib
            assert True
        except ImportError:
            pytest.fail("Could not import ngclearn or ngcsimlib")
    
    def test_context_creation(self):
        """Test ngclearn Context creation."""
        from ngclearn import Context
        
        with Context("test_context") as ctx:
            assert ctx is not None
    
    def test_component_creation(self):
        """Test creating ngclearn components."""
        from ngclearn import Context
        from ngclearn.components import RateCell, DenseSynapse
        
        with Context("test_components") as ctx:
            z1 = RateCell(name="z1", n_units=10, batch_size=1, tau_m=10.0)
            w1 = DenseSynapse(name="w1", shape=(10, 5), eta=0.01)
            
            assert z1 is not None
            assert w1 is not None
    
    def test_component_wiring(self):
        """Test wiring components together."""
        from ngclearn import Context
        from ngclearn.components import RateCell, DenseSynapse
        
        with Context("test_wiring") as ctx:
            z1 = RateCell(name="z1", n_units=10, batch_size=1, tau_m=10.0)
            z2 = RateCell(name="z2", n_units=5, batch_size=1, tau_m=10.0)
            w1 = DenseSynapse(name="w1", shape=(10, 5), eta=0.01)
            
            # Wire: z1 -> w1 -> z2
            w1.inputs << z1.zF
            z2.j << w1.outputs
            
            assert True
    
    def test_error_cell(self):
        """Test GaussianErrorCell functionality."""
        from ngclearn import Context
        from ngclearn.components import GaussianErrorCell
        import jax.numpy as jnp
        
        with Context("test_error") as ctx:
            error = GaussianErrorCell(
                name="error",
                n_units=5,
                batch_size=1,
                sigma=1.0
            )
            
            # Set values
            error.mu.set(jnp.ones((1, 5)))
            error.target.set(jnp.zeros((1, 5)))
            
            assert error is not None
    
    def test_process_compilation(self):
        """Test process compilation and execution."""
        from ngclearn import Context
        from ngclearn.components import RateCell
        from ngcsimlib.compilers.process import Process
        from jax import jit
        import jax.numpy as jnp
        
        with Context("test_process") as ctx:
            z1 = RateCell(name="z1", n_units=5, batch_size=1, tau_m=10.0)
            
            # Create and compile process
            advance_process = Process("advance") >> z1.advance_state
            ctx.wrap_and_add_command(jit(advance_process.pure), name="advance")
            
            reset_process = Process("reset") >> z1.reset
            ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
            
            # Execute
            ctx.reset()
            ctx.advance(t=0.0, dt=1.0)
            
            assert True
    
    def test_jax_functionality(self):
        """Test JAX operations work correctly."""
        import jax.numpy as jnp
        from jax import random
        
        key = random.PRNGKey(42)
        x = random.normal(key, (10, 5))
        
        # Test basic operations
        y = jnp.dot(x, jnp.ones((5, 3)))
        z = jnp.mean(jnp.square(y))
        
        assert jnp.isfinite(z)




