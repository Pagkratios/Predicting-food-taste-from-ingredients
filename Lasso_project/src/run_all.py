#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master runner for selective pipeline execution.

Usage:
    python3 Lasso_project/src/run_all.py
    python3 Lasso_project/src/run_all.py --steps plots --plot-groups gaussians combined
    python3 Lasso_project/src/run_all.py --steps train
    python3 Lasso_project/src/run_all.py --steps preprocess plots train --clean
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

STEP_MAP = {
    "preprocess": ("Preprocess", "preprocess.py"),
    "plots": ("EDA Plots", "data_plots.py"),
    "train": ("Train & Evaluate", "train.py"),
}


def wipe_outputs():
    """Remove all generated outputs for a clean start."""
    for d in (RESULTS_DIR, PROCESSED_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  Removed {d}")
    print()


def parse_args():
    """Parse command-line options for selective pipeline execution."""
    parser = argparse.ArgumentParser(description="Run selected Recipe Formulation pipeline steps.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEP_MAP.keys()),
        default=list(STEP_MAP.keys()),
        help="Pipeline steps to run. Default: preprocess plots train",
    )
    parser.add_argument(
        "--plot-groups",
        nargs="+",
        choices=["all", "ingredient_usage", "gaussians", "combined", "pca", "tsne", "clusters"],
        default=None,
        help="Forward selective plot groups to data_plots.py when plots is included.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated outputs before running selected steps.",
    )
    return parser.parse_args()


def run_step(name, script, extra_args=None):
    """Run a single pipeline step and abort on failure."""
    script_path = os.path.join(SRC_DIR, script)
    print(f"{'─' * 60}")
    print(f"  Step: {name}  ({script})")
    print(f"{'─' * 60}")

    t0 = time.time()
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] {name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[OK] {name} completed in {elapsed:.1f}s\n")


def main():
    args = parse_args()

    print("=" * 60)
    print("  Recipe Formulation - Pipeline Runner")
    print("=" * 60)
    print()

    if args.clean:
        print("[1/?] Cleaning selected output roots...")
        wipe_outputs()

    total_steps = len(args.steps)
    for i, step_key in enumerate(args.steps, start=1):
        name, script = STEP_MAP[step_key]
        print(f"[{i}/{total_steps}] {name}")
        extra_args = []
        if step_key == "plots" and args.plot_groups:
            extra_args.extend(["--only", *args.plot_groups])
        run_step(name, script, extra_args=extra_args)

    print("=" * 60)
    print("  All done. Results in Lasso_project/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
