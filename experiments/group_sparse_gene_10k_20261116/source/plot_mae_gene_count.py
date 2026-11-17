"""Plot the recorded 128-city MAE at four gene budgets; no model execution."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/learned_s42/city_confirmation_metrics.json"
    result = json.loads(source.read_text())
    rows = sorted(result["rates"].values(), key=lambda row: row["gene_cap"])
    assert [row["gene_cap"] for row in rows] == [8, 16, 32, 64]
    assert result["count"] == 128
    for row in rows:
        assert row["images"] == 128
        assert row["mean_active_genes"] == row["gene_cap"]
        if "per_image" in row:
            measured = sum(image["l1"] for image in row["per_image"]) / len(row["per_image"])
            assert len(row["per_image"]) == 128
            assert abs(measured - row["l1"]) < 1e-7

    output = root / "reports/mae_vs_gene_count_learned_s42"
    output.mkdir(exist_ok=True)
    table = [dict(active_genes=row["mean_active_genes"], mae=row["l1"], images=row["images"])
             for row in rows]
    with (output / "mae_vs_gene_count.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["active_genes", "mae", "images"])
        writer.writeheader()
        writer.writerows(table)
    metadata = dict(source=str(source), variant="learned_s42", evaluation_set="128-city diagnostic",
                    rgb_range=[0, 1], xscale="linear", yscale="linear", metric="mean absolute RGB error",
                    checkpoint_selection=result["selection_policy"], rows=table)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    fig.subplots_adjust(left=.12, right=.96, bottom=.19, top=.79)
    fig.suptitle("Reconstruction MAE vs. active genes", x=.12, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.12, .885, "Learned dictionary (seed 42)  |  128-city diagnostic  |  RGB in [0, 1]",
             fontsize=10, color="#525866")
    x = [row["active_genes"] for row in table]
    y = [row["mae"] for row in table]
    ax.plot(x, y, color="#2563eb", linewidth=2.4, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.5f}", (xi, yi), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=11, weight="medium", color="#1e3a8a")
    ax.set(xlabel="Active genes per image", ylabel="Mean absolute error (MAE)",
           xlim=(4, 68), ylim=(0, .05), xticks=[8, 16, 32, 64])
    ax.yaxis.set_major_locator(MultipleLocator(.01))
    ax.grid(axis="y", color="#dce2e9", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=4, color="#9ca3af")
    fig.text(.12, .045, "Mean over 128 images; lower is better. Same checkpoint for all four budgets.",
             fontsize=9, color="#525866")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"mae_vs_gene_count.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    print(json.dumps(dict(output=str(output), rows=table,
                          reduction_8_to_64_percent=100 * (y[0] - y[-1]) / y[0],
                          reduction_32_to_64_percent=100 * (y[2] - y[-1]) / y[2]), indent=2))


if __name__ == "__main__":
    main()
