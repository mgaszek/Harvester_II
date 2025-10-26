#!/usr/bin/env python3
"""
Development Environment Setup Script for Harvester II.
Sets up pre-commit hooks, installs development dependencies, and configures the development environment.
"""

from pathlib import Path
import subprocess  # nosec B404 - subprocess needed for development setup
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"[*] {description}...")
    try:
        # Split command into list for safer subprocess execution
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(  # nosec B603 - safe command execution in controlled environment
            cmd, check=True, capture_output=True, text=True
        )
        print(f"[+] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("[+] Harvester II Development Environment Setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(
            "[X] Error: pyproject.toml not found. Please run this script from the project root."
        )
        sys.exit(1)

    # Install development dependencies
    if not run_command(
        "pip install -e .", "Installing Harvester II in development mode"
    ):
        sys.exit(1)

    # Install development dependencies
    dev_deps = [
        "ruff>=0.6.0",
        "mypy>=1.11.0",
        "bandit>=1.7.0",
        "pre-commit>=3.0.0",
        "commitizen>=3.29.0",
    ]

    print(f"[*] Installing development dependencies: {', '.join(dev_deps)}")
    if not run_command(
        f"pip install {' '.join(dev_deps)}", "Installing development dependencies"
    ):
        print(
            "[!] Warning: Some development dependencies may not have installed correctly"
        )
        print("   You can install them manually with: pip install -r requirements.txt")

    # Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks"):
        print("[+] Pre-commit hooks installed successfully")
        print("   Hooks will run automatically on git commit")
    else:
        print("[!] Pre-commit hooks installation failed")
        print("   You can install them manually with: pre-commit install")

    # Run initial pre-commit checks
    print("\n[*] Running initial code quality checks...")
    if run_command(
        "pre-commit run --all-files", "Running pre-commit checks on all files"
    ):
        print("[+] All code quality checks passed!")
    else:
        print("[!] Some code quality checks failed")
        print("   You can run them individually:")
        print("   - ruff check . --fix")
        print("   - mypy src/")
        print("   - bandit -r src/")

    # Setup commitizen
    print("\n[*] Setting up commitizen for conventional commits...")
    if run_command("cz init", "Initializing commitizen"):
        print("[+] Commitizen initialized")
    else:
        print("[!] Commitizen initialization failed")
        print("   You can initialize it manually with: cz init")

    print("\n[+] Development environment setup complete!")
    print("\n[*] Next steps:")
    print("1. Review and customize .pre-commit-config.yaml if needed")
    print("2. Review pyproject.toml for linting and testing configuration")
    print("3. Run 'pre-commit run --all-files' to check all files")
    print("4. Use 'cz commit' for conventional commit messages")
    print("5. Run 'python src/benchmark_data_processing.py' to test data processing")
    print("6. Run 'pytest' to run the test suite")

    print("\n[*] Available commands:")
    print("- make format: Format code with ruff")
    print("- make lint: Run all linting checks")
    print("- make test: Run test suite")
    print("- make benchmark: Run performance benchmarks")
    print("- cz commit: Create conventional commit")


if __name__ == "__main__":
    main()
