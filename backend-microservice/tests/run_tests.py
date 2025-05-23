#!/usr/bin/env python3
"""
Simple test runner for getting started with testing.
"""

import subprocess
import sys


def run_tests():
    """Run the basic test suite."""
    print("🧪 Running basic tests...")

    # Run pytest with coverage
    cmd = [
        "poetry",
        "run",
        "pytest",
        "--cov=backend_microservice",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v",
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed.")

    return result.returncode


def run_basic_tests():
    """Run just the basic tests without coverage."""
    print("🧪 Running basic tests (no coverage)...")

    cmd = ["poetry", "run", "pytest", "-v"]

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        sys.exit(run_basic_tests())
    else:
        sys.exit(run_tests())
