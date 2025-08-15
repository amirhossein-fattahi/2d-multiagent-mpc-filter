import argparse, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_latest(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def smooth(s, w=10):
    return s.rolling(w, min_periods=1).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="Directory with CSV files")
    ap.add_argument("--baseline", default=None, help="Path to a specific baseline CSV (optional)")
    ap.add_argument("--filtered", default=None, help="Path to a specific filtered CSV (optional)")
    ap.add_argument("--window", type=int, default=10, help="Rolling window for smoothing")
    ap.add_argument("--save", action="store_true", help="Save plots as PNGs to results-dir/plots")
    args = ap.parse_args()

    # Auto-discover latest files if not provided
    baseline_csv = args.baseline or find_latest(os.path.join(args.results_dir, "baseline_*.csv"))
    filtered_csv = args.filtered or find_latest(os.path.join(args.results_dir, "filtered_*.csv"))

    if baseline_csv is None:
        raise FileNotFoundError(f"No baseline CSV found in {args.results_dir} (pattern baseline_*.csv)")
    if filtered_csv is None:
        raise FileNotFoundError(f"No filtered CSV found in {args.results_dir} (pattern filtered_*.csv)")

    print(f"Using baseline: {baseline_csv}")
    print(f"Using filtered: {filtered_csv}")

    base = pd.read_csv(baseline_csv)
    filt = pd.read_csv(filtered_csv)

    # --- Plot 1: Success (smoothed) ---
    plt.figure()
    plt.plot(base["episode"], smooth(base["success"], args.window), label="Baseline")
    plt.plot(filt["episode"], smooth(filt["success"], args.window), label="Filtered")
    plt.ylabel("Success (rolling mean)")
    plt.xlabel("Episode")
    plt.legend()
    plt.tight_layout()
    if args.save:
        outdir = os.path.join(args.results_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "success_rolling.png"), dpi=150)
    else:
        plt.show()

    # --- Plot 2: Collisions per episode ---
    plt.figure()
    plt.plot(base["episode"], base["collisions"], label="Baseline")
    plt.plot(filt["episode"], filt["collisions"], label="Filtered")
    plt.ylabel("Collisions")
    plt.xlabel("Episode")
    plt.legend()
    plt.tight_layout()
    if args.save:
        outdir = os.path.join(args.results_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "collisions.png"), dpi=150)
    else:
        plt.show()

    # --- Plot 3: Return (smoothed) ---
    if "return_mean" in base.columns and "return_mean" in filt.columns:
        plt.figure()
        plt.plot(base["episode"], smooth(base["return_mean"], args.window), label="Baseline")
        plt.plot(filt["episode"], smooth(filt["return_mean"], args.window), label="Filtered")
        plt.ylabel("Return (rolling mean)")
        plt.xlabel("Episode")
        plt.legend()
        plt.tight_layout()
        if args.save:
            outdir = os.path.join(args.results_dir, "plots")
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, "return_rolling.png"), dpi=150)
        else:
            plt.show()

    # --- Plot 4: Violation rate (smoothed) ---
    plt.figure()
    plt.plot(base["episode"], smooth((base["collisions"] > 0).astype(float), args.window), label="Baseline")
    plt.plot(filt["episode"], smooth((filt["collisions"] > 0).astype(float), args.window), label="Filtered")
    plt.ylabel("Violation rate (rolling)")
    plt.xlabel("Episode")
    plt.legend()
    plt.tight_layout()
    if args.save:
        outdir = os.path.join(args.results_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "violation_rate_rolling.png"), dpi=150)
    else:
        plt.show()

if __name__ == "__main__":
    main()
