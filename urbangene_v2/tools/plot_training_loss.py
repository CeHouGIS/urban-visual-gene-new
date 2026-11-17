#!/usr/bin/env python3
"""Plot recorded epoch losses without interpolation or smoothing."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import writable


def plot(run):
    run = writable(run)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "outputs/v2/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    records = [json.loads(line) for line in (run / "epochs.jsonl").read_text().splitlines()
               if line.strip()]
    records = sorted(records, key=lambda r: r["epoch"])
    if not records or [r["epoch"] for r in records] != list(range(1, records[-1]["epoch"] + 1)):
        raise ValueError("Expected one recorded result per completed epoch")
    epochs = [r["epoch"] for r in records]
    training = [r["train_reconstruction_mse"] for r in records]
    validation = [r["validation_reconstruction_mse"] for r in records]
    best = min(range(len(records)), key=validation.__getitem__)
    config = json.loads((run / "execution.json").read_text())
    planned = 30
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    ax.plot(epochs, training, "o-", color="#2563b3", linewidth=2, markersize=4,
            label="Training: mean across each epoch")
    ax.plot(epochs, validation, "o-", color="#dd7925", linewidth=2, markersize=4,
            label="Validation: evaluated at epoch end")
    ax.scatter([epochs[best]], [validation[best]], s=125, marker="*", color="#b65415", zorder=5)
    ax.annotate(f"Best validation: {validation[best]:.5f}\nEpoch {epochs[best]}",
                xy=(epochs[best], validation[best]), xytext=(epochs[best] - 5, validation[best] + .0055),
                arrowprops={"arrowstyle": "->", "color": "#666666"}, fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#dddddd"})
    ax.axvline(epochs[best], color="#b65415", linestyle=":", alpha=.45, linewidth=1)
    ax.set_title(f"UrbanGene | 20,000 real training images / 512 validation images\n"
                 f"Feature reconstruction loss | {epochs[-1]} of {planned} epochs completed",
                 loc="left", fontsize=13, pad=16)
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel("DINOv3 feature MSE (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.set_xlim(.5, max(epochs) + .5)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(.015, -.025, "Recorded epoch values; no smoothing. Incomplete epochs are excluded. "
             "This loss measures feature reconstruction, not instance segmentation accuracy.",
             fontsize=8, color="#555555")
    stem = run / f"loss_curve_epoch_{epochs[-1]:03d}"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "updates", "training_feature_mse", "validation_feature_mse"])
        writer.writerows(zip(epochs, [r["steps"] for r in records], training, validation))
    snapshot = {"run": str(run), "source": str(run / "epochs.jsonl"),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "completed_epochs": epochs[-1], "planned_epochs": planned,
                "best_epoch": epochs[best], "best_validation_mse": validation[best],
                "latest_training_mse": training[-1], "latest_validation_mse": validation[-1],
                "cpu_fallback": config["cpu_fallback"], "smoothing": False,
                "incomplete_epochs_included": False}
    stem.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({**snapshot, "plot": str(stem.with_suffix('.png'))}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    plot(Path(p.parse_args().run))
