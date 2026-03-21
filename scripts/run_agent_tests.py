#!/usr/bin/env python3
"""
GPUWorkLib Agent Test Runner
Runs Python tests for selected modules. Respects GPU (skip ROCm on NVIDIA, etc.).
Usage: python run_agent_tests.py [all | <module> | --file <path>]
"""
import subprocess
import sys
import os
from pathlib import Path

# Order for "all" (matches create_agent_test.md)
DEFAULT_ORDER = [
    "drvgpu",
    "fft_func",
    "statistics",
    "vector_algebra",
    "filters",
    "signal_generators",
    "lch_farrow",
    "heterodyne",
]

# Python test dirs (drvgpu has no Python tests)
MODULE_TO_PYTHON_DIR = {
    "drvgpu": None,
    "fft_func": "Python_test/fft_func",
    "statistics": "Python_test/statistics",
    "vector_algebra": "Python_test/vector_algebra",
    "filters": "Python_test/filters",
    "signal_generators": "Python_test/signal_generators",
    "lch_farrow": "Python_test/lch_farrow",
    "heterodyne": "Python_test/heterodyne",
}


def detect_gpu():
    """Detect GPU: amd or nvidia."""
    try:
        r = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and "Marketing" in (r.stdout or ""):
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        if r.returncode == 0:
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def read_modules_from_file(path):
    """Read modules from file. # = comment (full line or end-of-line)."""
    modules = []
    for p in (path, os.path.join("..", path)):
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                # Strip end-of-line comment
                if "#" in line:
                    line = line[: line.index("#")]
                line = line.strip()
                if line:
                    modules.append(line.lower())
        return modules
    return modules


def find_test_files(test_dir: str) -> list:
    """Find all test_*.py files in directory (non-recursive)."""
    p = Path(test_dir)
    if not p.exists():
        return []
    return sorted(p.glob("test_*.py"))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # PYTHONPATH: gpuworklib может быть в build/python или build/<config>/python
    for sub in ["build/python", "build/debian-radeon9070/python", "build/Release/python", "build/Debug/python"]:
        p = os.path.join(project_root, sub)
        if os.path.isdir(p):
            os.environ["PYTHONPATH"] = p + os.pathsep + os.environ.get("PYTHONPATH", "")
            break

    gpu = detect_gpu()
    print(f"GPU: {gpu}")
    if gpu == "nvidia":
        print("(Skipping ROCm-only modules: statistics, vector_algebra)")

    if len(sys.argv) < 2:
        print("Usage: python run_agent_tests.py [all | <module> | --file <path>]")
        sys.exit(1)

    arg1 = sys.argv[1].lower()
    if arg1 == "all":
        modules = read_modules_from_file("config/tests_order.txt")
        if not modules:
            modules = DEFAULT_ORDER.copy()
    elif arg1 == "--file" and len(sys.argv) >= 3:
        modules = read_modules_from_file(sys.argv[2])
        if not modules:
            print(f"No modules in file or not found: {sys.argv[2]}")
            sys.exit(1)
    else:
        modules = [arg1]

    failed = []
    total = 0
    for mod in modules:
        py_dir = MODULE_TO_PYTHON_DIR.get(mod)
        if not py_dir or not os.path.isdir(py_dir):
            continue

        if gpu == "nvidia" and mod in ("statistics", "vector_algebra"):
            print(f"  [SKIP] {mod} (ROCm-only)")
            continue

        test_files = find_test_files(py_dir)
        if not test_files:
            continue

        total += 1
        print(f"  >>> {mod} ({len(test_files)} files)")

        for test_file in test_files:
            r = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=project_root,
            )
            if r.returncode != 0:
                failed.append(f"{mod}/{test_file.name}")

    if failed:
        print(f"\n  Failed: {failed}")
        sys.exit(1)
    print(f"\n  Done: {total} module(s)")


if __name__ == "__main__":
    main()
