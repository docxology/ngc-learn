#!/usr/bin/env python3
"""
Verify NGC Inference installation.

Tests all critical imports and basic functionality.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        "jax",
        "jax.numpy",
        "numpy",
        "scipy",
        "matplotlib",
        "pytest",
        "yaml",
        "loguru",
        "ngclearn",
        "ngcsimlib",
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("✓ All imports successful\n")
    return True


def test_ngc_inference():
    """Test NGC Inference package."""
    print("Testing NGC Inference...")
    
    try:
        from ngc_inference import __version__
        print(f"  ✓ ngc_inference version: {__version__}")
        
        from ngc_inference.core.free_energy import compute_free_energy
        print("  ✓ Free energy module")
        
        from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
        print("  ✓ Simple prediction agent")
        
        from ngc_inference.simulations.hierarchical_inference import HierarchicalInferenceAgent
        print("  ✓ Hierarchical inference agent")
        
        from ngc_inference.orchestrators.simulation_runner import SimulationRunner
        print("  ✓ Simulation runner")
        
        from ngc_inference.utils.logging_config import get_logger
        print("  ✓ Logging utilities")
        
        print("✓ NGC Inference package verified\n")
        return True
        
    except Exception as e:
        print(f"❌ NGC Inference test failed: {e}\n")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")
    
    try:
        import jax.numpy as jnp
        from jax import random
        from ngc_inference.core.free_energy import compute_free_energy
        
        # Test free energy computation
        key = random.PRNGKey(42)
        obs = random.normal(key, (1, 10))
        pred = random.normal(key, (1, 10))
        prior = jnp.zeros((1, 5))
        posterior = random.normal(key, (1, 5))
        
        fe, components = compute_free_energy(obs, pred, prior, posterior)
        
        assert jnp.isfinite(fe), "Free energy is not finite"
        print(f"  ✓ Free energy computation: {float(fe):.4f}")
        
        # Test simple agent
        from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
        
        agent = SimplePredictionAgent(
            n_observations=10,
            n_hidden=5,
            seed=42
        )
        
        beliefs, metrics = agent.infer(obs, n_steps=5)
        
        assert beliefs.shape == (1, 5), "Beliefs have wrong shape"
        assert "free_energy" in metrics, "Metrics missing free energy"
        print(f"  ✓ Agent inference: FE={metrics['free_energy']:.4f}")
        
        print("✓ Basic functionality verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_ngclearn_components():
    """Test that ngclearn components work."""
    print("Testing ngclearn components...")
    
    try:
        from ngclearn import Context
        from ngclearn.components import RateCell, GaussianErrorCell, DenseSynapse
        import jax.numpy as jnp
        
        with Context("test") as ctx:
            z1 = RateCell(name="z1", n_units=5, batch_size=1, tau_m=10.0)
            error = GaussianErrorCell(name="e1", n_units=5, batch_size=1, sigma=1.0)
            synapse = DenseSynapse(name="w1", shape=(5, 5), eta=0.01)
        
        print("  ✓ RateCell created")
        print("  ✓ GaussianErrorCell created")
        print("  ✓ DenseSynapse created")
        print("✓ ngclearn components verified\n")
        return True
        
    except Exception as e:
        print(f"❌ ngclearn components test failed: {e}\n")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("NGC Inference Installation Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("NGC Inference Package", test_ngc_inference),
        ("ngclearn Components", test_ngclearn_components),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print()
    if all_passed:
        print("🎉 All verification tests passed!")
        print("Installation is successful and ready to use.")
        return 0
    else:
        print("⚠️  Some verification tests failed.")
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())




