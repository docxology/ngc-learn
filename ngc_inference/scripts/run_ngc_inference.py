#!/usr/bin/env python3
"""
Comprehensive NGC Inference runner.

Runs complete setup, verification, and all examples with logging.
Confirms all functionality works correctly.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import platform


class NGCInferenceRunner:
    """Comprehensive runner for NGC Inference setup and examples."""
    
    def __init__(self):
        """Initialize runner."""
        self.root_dir = Path(__file__).parent.parent
        self.log_file = self.root_dir / "logs" / f"run_ngc_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine virtual environment paths
        self.venv_dir = self.root_dir / ".venv"
        if platform.system() == "Windows":
            self.venv_python = self.venv_dir / "Scripts" / "python.exe"
            self.venv_pytest = self.venv_dir / "Scripts" / "pytest.exe"
        else:
            self.venv_python = self.venv_dir / "bin" / "python"
            self.venv_pytest = self.venv_dir / "bin" / "pytest"
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "root_dir": str(self.root_dir),
            "venv_dir": str(self.venv_dir),
            "steps": {}
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {level}: {message}"
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def get_python_cmd(self):
        """Get the Python command to use (venv if available, else system python)."""
        if self.venv_python.exists():
            return str(self.venv_python)
        else:
            self.log("Virtual environment not found, using system python3", "WARNING")
            return "python3"
    
    def get_pytest_cmd(self):
        """Get the pytest command to use (venv if available, else system pytest)."""
        if self.venv_pytest.exists():
            return str(self.venv_pytest)
        else:
            self.log("Virtual environment pytest not found, using system pytest", "WARNING")
            return "pytest"
    
    def run_command(self, cmd: list, description: str, cwd=None, use_shell=False) -> bool:
        """
        Run shell command and capture output.
        
        Args:
            cmd: Command and arguments (list) or command string (if use_shell=True)
            description: Description of command
            cwd: Working directory
            use_shell: Whether to run command through shell
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"Running: {description}")
        if isinstance(cmd, list):
            self.log(f"Command: {' '.join(str(c) for c in cmd)}")
        else:
            self.log(f"Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.root_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                shell=use_shell
            )
            
            if result.stdout:
                self.log(result.stdout, "OUTPUT")
            
            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully", "SUCCESS")
                return True
            else:
                self.log(f"✗ {description} failed with code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(result.stderr, "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ {description} timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"✗ {description} failed: {e}", "ERROR")
            return False
    
    def step_environment_setup(self) -> bool:
        """Step 1: Setup environment."""
        self.log("=" * 70)
        self.log("STEP 1: Environment Setup")
        self.log("=" * 70)
        
        setup_script = self.root_dir / "scripts" / "setup_environment.sh"
        
        if not setup_script.exists():
            self.log("Setup script not found, attempting manual setup", "WARNING")
            
            # Check if we're on Windows
            if platform.system() == "Windows":
                self.log("Windows detected - manual setup required", "ERROR")
                self.log("Please run: pip install -e .[dev,docs]", "ERROR")
                success = False
            else:
                # Try manual pip install with proper order
                self.log("Installing numpy first...")
                numpy_success = self.run_command(
                    ["pip", "install", "numpy>=1.22.0"],
                    "Install numpy"
                )
                if numpy_success:
                    self.log("Upgrading scikit-image...")
                    self.run_command(
                        ["pip", "install", "scikit-image>=0.21.0", "--upgrade"],
                        "Upgrade scikit-image"
                    )
                    # Continue even if scikit-image upgrade fails (non-critical)
                    self.log("Installing NGC Inference...")
                    success = self.run_command(
                        ["pip", "install", "-e", ".[dev,docs]"],
                        "Manual pip installation"
                    )
                else:
                    success = False
        else:
            # Run setup script
            success = self.run_command(
                ["bash", str(setup_script)],
                "Environment setup",
                use_shell=False
            )
        
        # Verify venv was created
        if success and self.venv_python.exists():
            self.log(f"Virtual environment created at: {self.venv_dir}", "SUCCESS")
            self.log(f"Using Python: {self.venv_python}", "INFO")
        elif not self.venv_python.exists():
            self.log("Virtual environment not found, will use system Python", "WARNING")
        
        self.results["steps"]["environment_setup"] = {
            "success": success,
            "venv_created": self.venv_python.exists(),
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def step_verify_installation(self) -> bool:
        """Step 2: Verify installation."""
        self.log("=" * 70)
        self.log("STEP 2: Verify Installation")
        self.log("=" * 70)
        
        verify_script = self.root_dir / "scripts" / "verify_installation.py"
        
        if not verify_script.exists():
            self.log("Verification script not found", "ERROR")
            return False
        
        python_cmd = self.get_python_cmd()
        success = self.run_command(
            [python_cmd, str(verify_script)],
            "Installation verification"
        )
        
        self.results["steps"]["verify_installation"] = {
            "success": success,
            "python_used": python_cmd,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def step_run_simple_example(self) -> bool:
        """Step 3: Run simple prediction example."""
        self.log("=" * 70)
        self.log("STEP 3: Run Simple Prediction Example")
        self.log("=" * 70)
        
        simple_script = self.root_dir / "scripts" / "run_simple_example.py"
        
        if not simple_script.exists():
            self.log("Simple example script not found", "ERROR")
            return False
        
        python_cmd = self.get_python_cmd()
        success = self.run_command(
            [python_cmd, str(simple_script)],
            "Simple prediction example"
        )
        
        self.results["steps"]["simple_example"] = {
            "success": success,
            "python_used": python_cmd,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def step_run_hierarchical_example(self) -> bool:
        """Step 4: Run hierarchical inference example."""
        self.log("=" * 70)
        self.log("STEP 4: Run Hierarchical Inference Example")
        self.log("=" * 70)
        
        hierarchical_script = self.root_dir / "scripts" / "run_hierarchical_example.py"
        
        if not hierarchical_script.exists():
            self.log("Hierarchical example script not found", "ERROR")
            return False
        
        python_cmd = self.get_python_cmd()
        success = self.run_command(
            [python_cmd, str(hierarchical_script)],
            "Hierarchical inference example"
        )
        
        self.results["steps"]["hierarchical_example"] = {
            "success": success,
            "python_used": python_cmd,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def step_run_tests(self) -> bool:
        """Step 5: Run test suite."""
        self.log("=" * 70)
        self.log("STEP 5: Run Test Suite")
        self.log("=" * 70)
        
        # Use venv pytest if available
        pytest_cmd = self.get_pytest_cmd()
        
        # Run from the tests directory
        tests_dir = self.root_dir / "tests"
        if not tests_dir.exists():
            self.log("Tests directory not found", "ERROR")
            return False
        
        success = self.run_command(
            [pytest_cmd, str(tests_dir), "-v", "--tb=short"],
            "Test suite execution"
        )
        
        self.results["steps"]["tests"] = {
            "success": success,
            "pytest_used": pytest_cmd,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def generate_summary_report(self):
        """Generate final summary report."""
        self.log("=" * 70)
        self.log("SUMMARY REPORT")
        self.log("=" * 70)
        
        total_steps = len(self.results["steps"])
        successful_steps = sum(1 for s in self.results["steps"].values() if s["success"])
        
        self.results["summary"] = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0
        }
        
        self.log(f"Total Steps: {total_steps}")
        self.log(f"Successful: {successful_steps}")
        self.log(f"Failed: {total_steps - successful_steps}")
        self.log(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        
        self.log("\nStep Details:")
        for step_name, step_data in self.results["steps"].items():
            status = "✓ PASS" if step_data["success"] else "✗ FAIL"
            self.log(f"  {status}: {step_name}")
        
        # Save results to JSON
        results_file = self.root_dir / "logs" / f"run_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved to: {results_file}")
        self.log(f"Log file: {self.log_file}")
        
        return self.results["summary"]["failed_steps"] == 0
    
    def run_all(self) -> bool:
        """Run all steps in sequence."""
        self.log("=" * 70)
        self.log("NGC INFERENCE COMPREHENSIVE RUNNER")
        self.log("=" * 70)
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Working directory: {self.root_dir}")
        self.log(f"Virtual environment: {self.venv_dir}")
        self.log(f"Python: {self.venv_python if self.venv_python.exists() else 'system python3'}")
        self.log(f"Log file: {self.log_file}")
        self.log("")
        
        steps = [
            ("Environment Setup", self.step_environment_setup),
            ("Verify Installation", self.step_verify_installation),
            ("Simple Example", self.step_run_simple_example),
            ("Hierarchical Example", self.step_run_hierarchical_example),
            ("Test Suite", self.step_run_tests),
        ]
        
        for step_name, step_func in steps:
            try:
                success = step_func()
                if not success:
                    self.log(f"Step '{step_name}' failed, continuing...", "WARNING")
            except Exception as e:
                self.log(f"Step '{step_name}' crashed: {e}", "ERROR")
            
            self.log("")  # Blank line between steps
        
        # Generate summary
        final_success = self.generate_summary_report()
        
        self.log("")
        self.log("=" * 70)
        if final_success:
            self.log("🎉 ALL STEPS COMPLETED SUCCESSFULLY!", "SUCCESS")
            self.log("NGC Inference is fully functional and verified.", "SUCCESS")
        else:
            self.log("⚠️  SOME STEPS FAILED", "WARNING")
            self.log("Please review the log file for details.", "WARNING")
        self.log("=" * 70)
        
        return final_success


def main():
    """Main entry point."""
    runner = NGCInferenceRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



