"""
Plot training curves for saved Lunar Lander runs (console script).

This reads the JSON stats in the `stats/` folder and reproduces
score / moving‑average graphs similar to the project report.

Usage examples (from project root):

  python plot_training.py                 # auto-pick first model from descriptions.txt
  python plot_training.py --model m5      # pick specific stats key (e.g. m0..m7)
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib

# Use a non-GUI backend so we don't depend on system GUI/image resources.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


STATS_DIR = "stats"
DESCRIPTIONS_FILE = os.path.join(STATS_DIR, "descriptions.txt")


def ma(x: List[float], w: int) -> np.ndarray:
    """Simple moving average with window w."""
    if len(x) < w:
        return np.array([])
    return np.convolve(np.asarray(x, dtype=float), np.ones(w), "valid") / float(w)


def load_models() -> Dict[str, Dict]:
    """Load model descriptions and score/eps histories from stats/."""
    models: Dict[str, Dict] = {}

    if not os.path.exists(DESCRIPTIONS_FILE):
        raise FileNotFoundError(
            f"Could not find '{DESCRIPTIONS_FILE}'. "
            "Make sure you run this from the project root and that stats/ exists."
        )

    with open(DESCRIPTIONS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines without a ';' separator
            if (not line) or (";" not in line):
                continue
            key, desc = line.split(";", 1)
            dd = desc.split()
            models[key] = {
                "desc": desc,
                "desclong": {
                    "ML Library": dd[0] if len(dd) > 0 else "",
                    "Algorithm": dd[1] if len(dd) > 1 else "",
                    "Transitions stored": dd[2] if len(dd) > 2 else "",
                    "Q network(s)": dd[3] if len(dd) > 3 else "",
                    "Batch size": dd[4] if len(dd) > 4 else "",
                    "Episodes": dd[5] if len(dd) > 5 else "",
                    "Update weights": dd[6] if len(dd) > 6 else "",
                    "Replace target Q weights": dd[7] if len(dd) > 7 else "",
                },
            }

    # Load the numeric histories
    for key in list(models.keys()):
        score_path = os.path.join(STATS_DIR, f"{key}score.json")
        if not os.path.exists(score_path):
            # If the score file does not exist, drop this model
            models.pop(key)
            continue
        with open(score_path, "r") as f:
            models[key]["score"] = json.load(f)

        eps_path = os.path.join(STATS_DIR, f"{key}eps.json")
        if os.path.exists(eps_path):
            with open(eps_path, "r") as f:
                models[key]["eps"] = json.load(f)

    if not models:
        raise RuntimeError("No models with score histories found under stats/.")

    return models


def plot_single_model(key: str, model: Dict):
    """Plot raw scores and MA100 for a single model."""
    scores = model["score"]
    n = len(scores)
    xs = np.arange(n)

    # Configure seaborn style to roughly match the notebook
    sns.set()

    plt.figure(figsize=(12, 6))

    # Raw episode scores (light gray)
    plt.plot(xs, scores, color="lightgray", linewidth=1, label="Score")

    # Moving average over last 100 episodes
    window = 100
    ma_vals = ma(scores, window)
    if ma_vals.size > 0:
        ma_x = np.arange(window - 1, n)

        # Find first index where MA crosses 200, if any
        over = np.where(ma_vals >= 200)[0]
        if over.size > 0:
            first_over = over[0]
            # Before 200 mark
            plt.plot(
                ma_x[: first_over + 1],
                ma_vals[: first_over + 1],
                color="blue",
                label="Score MA100 before 200 mark",
            )
            # After 200 mark
            plt.plot(
                ma_x[first_over:],
                ma_vals[first_over:],
                color="red",
                label="Score MA100 after 200 mark",
            )
            # Vertical line at the convergence point
            conv_step = ma_x[first_over]
            plt.axvline(conv_step, color="red", linestyle="--", alpha=0.5)
            plt.text(
                conv_step,
                210,
                f"Broke 200 point barrier: step {conv_step}",
                color="red",
                fontsize=9,
                rotation=90,
                va="bottom",
            )
        else:
            plt.plot(
                ma_x,
                ma_vals,
                color="blue",
                label="Score MA100",
            )

    # Pretty title showing configuration
    title_lines = [f"Model '{key}'", model.get("desc", "")]
    plt.title("\n".join(title_lines))
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.axhline(200, color="green", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Save to file instead of opening a window (works even without GUI backend)
    out_name = f"training_plot_{key}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved plot to '{out_name}'")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from stats/*.json.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key from stats/descriptions.txt (e.g. m0, m5). "
        "If omitted, the first available model is used.",
    )
    args = parser.parse_args()

    models = load_models()

    if args.model is None:
        # Just take the first key in sorted order
        key = sorted(models.keys())[0]
    else:
        key = args.model
        if key not in models:
            raise SystemExit(
                f"Model '{key}' not found. Available keys: {', '.join(sorted(models.keys()))}"
            )

    print(f"Plotting training curves for model '{key}'")
    plot_single_model(key, models[key])


if __name__ == "__main__":
    main()

