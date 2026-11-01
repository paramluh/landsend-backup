import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np


OBJECTIVES = ("stabilizer", "resolver")
LOSSES = ("velocity", "endpoint")
COLORS = {"velocity": "#3977b8", "endpoint": "#df7836"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gradient-label", default="Parameter gradient L2 norm (batch-mean loss)")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No probe records to plot")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    grouped = {objective: [r for r in records if r["objective"] == objective] for objective in OBJECTIVES}
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, objective in enumerate(OBJECTIVES):
        rows = grouped[objective]
        summary[objective] = {
            "n_microbatches": len(rows),
            "n_training_samples": sum(r["batch_size"] for r in rows),
        }
        for row, metric in enumerate(("loss", "grad_norm")):
            ax = axes[row, col]
            all_values = np.array([r[f"{loss}_{metric}"] for r in rows for loss in LOSSES], dtype=float)
            if len(all_values) and (not np.isfinite(all_values).all() or min(all_values) <= 0):
                raise ValueError(f"Log histograms require positive finite {objective} {metric}")
            if len(all_values):
                lower, upper = all_values.min(), all_values.max()
                if lower == upper:
                    lower, upper = lower / 2, upper * 2
                bins = np.geomspace(lower * 0.999, upper * 1.001, 9)
            for loss in LOSSES:
                values = np.array([r[f"{loss}_{metric}"] for r in rows], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {objective} {loss}_{metric}")
                summary[objective][f"{loss}_{metric}"] = {
                    "mean": float(values.mean()) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                }
                if len(values):
                    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.25, color=COLORS[loss])
                    ax.hist(values, bins=bins, histtype="step", linewidth=1.8, color=COLORS[loss], label=f"{loss.capitalize()} · median {np.median(values):.3g}")
                    ax.axvline(np.median(values), color=COLORS[loss], alpha=0.5, linestyle=":")
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{objective.capitalize()} · {len(rows)} microbatches")
            ax.set_xlabel("Loss" if metric == "loss" else args.gradient_label)
            ax.set_ylabel("Microbatch count")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Flow-matching scale probe: velocity vs endpoint loss", fontsize=14)
    fig.savefig(args.output_dir / "loss_gradient_histograms.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    objective_colors = {"stabilizer": "#3977b8", "resolver": "#a4519d"}
    for ax, loss in zip(axes, LOSSES):
        all_norms = []
        for objective in OBJECTIVES:
            rows = grouped[objective]
            times = np.array([r["t"] for r in rows], dtype=float)
            norms = np.array([r[f"{loss}_grad_norm"] for r in rows], dtype=float)
            ax.scatter(times, norms, color=objective_colors[objective], label=objective.capitalize(), s=38, alpha=0.75)
            for record, norm in zip(rows, norms):
                ax.hlines(norm, min(record["times"]), max(record["times"]), color=objective_colors[objective], alpha=0.3, linewidth=1)
            all_norms.extend(norms)
        if all_norms and min(all_norms) > 0:
            ax.set_yscale("log")
        ax.set_title(f"{loss.capitalize()} loss")
        ax.set_xlabel("Microbatch mean diffusion time")
        ax.set_ylabel(args.gradient_label)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Gradient scale versus diffusion time · bars span each microbatch’s sampled times", fontsize=12)
    fig.savefig(args.output_dir / "gradient_norm_vs_time.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
