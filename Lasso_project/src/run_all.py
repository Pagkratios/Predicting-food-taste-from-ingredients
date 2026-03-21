#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master runner: clean slate -> preprocess -> EDA plots -> train + evaluate.

Usage:
    python3 Lasso_project/src/run_all.py
"""

import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

STEPS = [
    ("Preprocess", "preprocess.py"),
    ("EDA Plots", "data_plots.py"),
    ("Train & Evaluate", "train.py"),
]


def wipe_outputs():
    """Remove all generated outputs for a clean start."""
    for d in (RESULTS_DIR, PROCESSED_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  Removed {d}")
    print()


def run_step(name, script):
    """Run a single pipeline step and abort on failure."""
    script_path = os.path.join(SRC_DIR, script)
    print(f"{'─' * 60}")
    print(f"  Step: {name}  ({script})")
    print(f"{'─' * 60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] {name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[OK] {name} completed in {elapsed:.1f}s\n")


def main():
    print("=" * 60)
    print("  Recipe Formulation - Full Pipeline")
    print("=" * 60)
    print()

    print("[1/4] Cleaning previous outputs...")
    wipe_outputs()

    for i, (name, script) in enumerate(STEPS, start=2):
        print(f"[{i}/{len(STEPS) + 1}] {name}")
        run_step(name, script)

    print("=" * 60)
    print("  All done. Results in Lasso_project/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
