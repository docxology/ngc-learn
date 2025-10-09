# NGC Inference Test Suite

Comprehensive test suite for NGC Inference using pytest with real ngclearn integration.

## Structure

```
tests/
├── unit/              # Unit tests for individual functions
│   ├── test_free_energy.py
│   └── test_metrics.py
├── integration/       # Integration tests for agents
│   ├── test_simple_agent.py
│   └── test_hierarchical_agent.py
├── simulations/       # End-to-end simulation tests
│   └── test_complete_workflow.py
└── test_ngclearn_integration.py  # Real ngclearn verification
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
Test individual functions in isolation:
- Free energy computations
- Prediction error calculations
- Expected free energy
- Entropy and KL divergence
- Metrics (RMSE, MAE, R²)

**Coverage**: Core mathematical operations and utility functions

### Integration Tests (`@pytest.mark.integration`)
Test agent functionality with ngclearn components:
- Agent initialization
- Inference execution
- Learning workflows
- Free energy minimization over time
- Real ngclearn component creation and wiring

**Coverage**: Complete agent behavior and ngclearn integration

### Simulation Tests (`@pytest.mark.simulation`)
Test end-to-end workflows:
- Complete simulation runs
- Orchestrator functionality
- Configuration management
- Result saving and loading
- Parameter sweep experiments

**Coverage**: Full system integration with file I/O

## Running Tests

### All Tests
```bash
cd ngc_inference
pytest tests/ -v
```

### By Category
```bash
pytest tests/unit/ -m unit -v
pytest tests/integration/ -m integration -v  
pytest tests/simulations/ -m simulation -v
```

### Specific Test File
```bash
pytest tests/unit/test_free_energy.py -v
pytest tests/integration/test_simple_agent.py -v
```

### With Coverage
```bash
pytest tests/ --cov=src/ngc_inference --cov-report=html
open htmlcov/index.html
```

### Fast Tests Only (skip slow)
```bash
pytest tests/ -v -m "not slow"
```

## Test Fixtures

Common fixtures defined in test files:
- `self.key`: JAX random key (PRNGKey(42))
- `self.n_features`: Feature dimensionality
- `self.batch_size`: Batch size for arrays
- `self.temp_dir`: Temporary directory for outputs

## Assertions

Standard assertions used:
- `assert`: Boolean conditions
- `assert_allclose()`: Numerical equality with tolerance
- `pytest.fail()`: Explicit test failure
- `pytest.raises()`: Exception testing

## Test Examples

### Unit Test Example
```python
@pytest.mark.unit
def test_prediction_error():
    observation = jnp.ones((1, 10))
    prediction = jnp.zeros((1, 10))
    precision = 1.0
    
    error = compute_prediction_error(observation, prediction, precision)
    
    assert error.shape == (1, 10)
    assert jnp.allclose(error, precision * (observation - prediction))
```

### Integration Test Example
```python
@pytest.mark.integration
def test_agent_initialization():
    agent = SimplePredictionAgent(
        n_observations=10,
        n_hidden=20,
        seed=42
    )
    
    assert agent.n_observations == 10
    assert agent.n_hidden == 20
    assert agent.context is not None
```

### Simulation Test Example
```python
@pytest.mark.simulation
def test_complete_workflow():
    config = {"agent": {"n_observations": 10, "n_hidden": 5}}
    agent = SimplePredictionAgent(**config["agent"])
    runner = SimulationRunner(config, output_dir=temp_dir)
    
    results = runner.run_learning(agent, data, n_epochs=5)
    
    assert "losses" in results
    assert (Path(temp_dir) / "config.yaml").exists()
```

## Test Requirements

All tests must:
1. **Be Independent**: No test depends on another
2. **Be Deterministic**: Same seed → same results
3. **Clean Up**: Remove temporary files/directories
4. **Be Fast**: Unit tests < 1s, integration < 10s
5. **Document Intent**: Clear docstrings explaining what's tested
6. **Use Real Data**: No mocks, always actual computations

## Continuous Integration

Tests run automatically on:
- Every commit (local pre-commit hook)
- Pull requests (GitHub Actions)
- Before releases (manual verification)

**CI Configuration**:
```yaml
# .github/workflows/tests.yml
- pytest tests/ -v --cov=src/ngc_inference
- pytest tests/ -v --cov-report=xml
- codecov upload
```

## Coverage Goals

Target coverage by module:
- **core/**: 95%+ (critical mathematical functions)
- **simulations/**: 90%+ (agent implementations)
- **orchestrators/**: 85%+ (workflow management)
- **utils/**: 85%+ (helper functions)

Overall target: **90%+ coverage**

## Test Data

Test data generation:
- **Random**: `random.normal(key, shape)` with fixed seed
- **Structured**: Sinusoidal, hierarchical patterns
- **Small Scale**: Quick execution (n_samples ≤ 50)
- **Representative**: Covers typical use cases

## Debugging Tests

When tests fail:
1. **Read Error Message**: pytest provides detailed tracebacks
2. **Run Single Test**: `pytest tests/unit/test_free_energy.py::TestFreeEnergy::test_prediction_error -v`
3. **Add Print Statements**: Temporarily for debugging
4. **Use Debugger**: `pytest --pdb` drops into debugger on failure
5. **Check Logs**: Review logs/ directory for simulation tests

## Writing New Tests

Template for new test:
```python
import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.module import function

@pytest.mark.unit  # or integration, simulation
class TestMyFeature:
    """Test suite for my feature."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        # Add fixtures
    
    def test_basic_functionality(self):
        """Test that basic functionality works."""
        # Arrange
        input_data = jnp.zeros((1, 10))
        
        # Act
        result = function(input_data)
        
        # Assert
        assert result.shape == (1, 10)
        assert jnp.all(jnp.isfinite(result))
```

## Performance Testing

Benchmark tests with `pytest-benchmark`:
```python
def test_inference_performance(benchmark):
    agent = SimplePredictionAgent(n_observations=10, n_hidden=20)
    obs = random.normal(key, (1, 10))
    
    result = benchmark(agent.infer, obs, n_steps=30)
    assert result is not None
```

Run benchmarks: `pytest tests/ --benchmark-only`

## Test Philosophy

1. **TDD**: Write tests before implementation
2. **Comprehensive**: Cover happy path + edge cases  
3. **Real Data**: Never mock ngclearn components
4. **Fast Feedback**: Quick tests for rapid iteration
5. **Maintainable**: Clear, simple, well-documented tests
6. **Isolated**: Tests don't affect each other
7. **Repeatable**: Deterministic with fixed seeds

## Common Patterns

### Testing with ngclearn
```python
from ngclearn import Context
from ngclearn.components import RateCell

with Context("test") as ctx:
    neuron = RateCell(name="z", n_units=10, batch_size=1, tau_m=10.0)
    assert neuron is not None
```

### Testing Free Energy Minimization
```python
beliefs, metrics = agent.infer(observation, n_steps=50)
fe_traj = metrics["free_energy_trajectory"]
assert fe_traj[-1] < fe_traj[0]  # Should decrease
```

### Testing File I/O
```python
import tempfile
temp_dir = tempfile.mkdtemp()
runner = SimulationRunner(config, output_dir=temp_dir)
# ... run simulation ...
assert (Path(temp_dir) / "results.json").exists()
```

## Troubleshooting

**Common Issues**:
- **Import Errors**: Check PYTHONPATH includes src/
- **JAX Errors**: Ensure JAX installed correctly
- **ngclearn Errors**: Verify ngclearn ≥ 2.0.3
- **File Not Found**: Check working directory
- **Random Variation**: Set seeds consistently

**Solutions**:
- Run `python scripts/verify_installation.py` first
- Check `pytest --collect-only` to see discovered tests
- Use `-v` for verbose output
- Add `--tb=short` for concise tracebacks

## Quality Metrics

Track test quality:
- **Pass Rate**: Should be 100%
- **Coverage**: Should be ≥ 90%
- **Execution Time**: Unit tests < 30s total
- **Flakiness**: Zero flaky tests
- **Maintenance**: Updated with code changes






