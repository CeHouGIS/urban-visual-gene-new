"""Compare saved, checkpoint-verified reconstructions on identical city inputs.

CPU plotting only; neither checkpoint is changed and no model is executed.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def measure(originals, reconstructions):
    delta = reconstructions.astype(np.float64) - originals.astype(np.float64)
    mae = np.abs(delta).mean(axis=(1, 2, 3))
    mse = np.square(delta).mean(axis=(1, 2, 3))
    return mae, -10 * np.log10(np.maximum(mse, 1e-12))


def main():
    root = Path(__file__).resolve().parents[1]
    old_root = root.parent / "residual_ista_gene_10k_scratch_20261116"
    old_reports = root.parent / "gene_validation_three_tests_20261116/reports/frozen_diagnostics"
    current_reports = root / "reports/scale_gene_views_20261117_042541"
    verification = json.loads((old_reports / "checkpoint_verification.json").read_text())
    assert verification["all_state_tensors_equal_checkpoint"]
    assert verification["checkpoint_step"] == 6000
    current_metadata = json.loads((current_reports / "metadata.json").read_text())
    assert current_metadata["variant"] == "learned_s42"
    assert current_metadata["checkpoint_step"] == 7500
    assert current_metadata["primary_gene_cap"] == 64
    assert current_metadata["image_indices"] == [0, 1, 2]

    old_source = old_reports / "confirmation_reconstruction.npz"
    current_source = current_reports / "scale_gene_arrays.npz"
    with np.load(old_source) as arrays:
        old_inputs = arrays["originals"]
        old_outputs = arrays["reconstructions"]
    with np.load(current_source) as arrays:
        inputs = arrays["original"]
        outputs = arrays["full_reconstruction"]
    assert inputs.shape == outputs.shape == (3, 3, 256, 256)
    assert old_inputs.shape == old_outputs.shape == (128, 3, 256, 256)
    assert np.array_equal(inputs, old_inputs[:3]), "Comparison requires identical input pixels."
    assert np.isfinite(old_outputs).all() and np.isfinite(outputs).all()
    old_mae, old_psnr = measure(inputs, old_outputs[:3])
    current_mae, current_psnr = measure(inputs, outputs)
    old_all_mae, old_all_psnr = measure(old_inputs, old_outputs)
    current_metrics = json.loads((root / "runs/learned_s42/city_confirmation_metrics.json").read_text())
    rate64 = current_metrics["rates"]["3"]
    assert rate64["gene_cap"] == 64 and rate64["images"] == 128

    destination = root / "reports/old_vs_current_same_streets_20261117"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 12.1))
    fig.subplots_adjust(left=.12, right=.985, bottom=.09, top=.86, hspace=.23, wspace=.05)
    measurements = []
    for row, sample in enumerate(current_metadata["images"]):
        for column, image_array in enumerate((inputs[row], old_outputs[row], outputs[row])):
            ax = axes[row, column]
            ax.imshow(image_array.transpose(1, 2, 0).clip(0, 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(sample["city"] + "\n" + sample["country"])
        axes[row, 1].set_xlabel(f"MAE {old_mae[row]:.5f} | PSNR {old_psnr[row]:.2f} dB", color="#14643c")
        axes[row, 2].set_xlabel(f"MAE {current_mae[row]:.5f} | PSNR {current_psnr[row]:.2f} dB", color="#a23b21")
        measurements.append({
            "city_index": row, "image_id": sample["id"], "city": sample["city"],
            "old": {"mae": float(old_mae[row]), "psnr": float(old_psnr[row])},
            "current_cap64": {"mae": float(current_mae[row]), "psnr": float(current_psnr[row])},
        })
    for column, title in enumerate(("Original", "Earlier multiscale ISTA\ncheckpoint 6000", "Current 64-gene model\nlearned_s42, checkpoint 7500")):
        axes[0, column].set_title(title, fontsize=12, pad=10)
    fig.suptitle("Different models: reconstruction quality on identical street inputs", fontsize=16, weight="bold", y=.975)
    fig.text(.12, .935,
             f"Same 128-city diagnostic: MAE {old_all_mae.mean():.5f} → {rate64['l1']:.5f}; "
             f"PSNR {old_all_psnr.mean():.2f} → {rate64['psnr_mean_per_image']:.2f} dB", fontsize=11)
    fig.text(.12, .908, "Displayed examples are the fixed first three city inputs; pixels match exactly.", fontsize=10, color="#4b5563")
    fig.text(.12, .045, "Capacity differs: old model has no 64-gene cap and uses higher-resolution float expression maps.", fontsize=9, color="#4b5563")
    fig.text(.12, .025, "Current model also changes the encoder and quantizes expressions. This does not isolate any one cause.", fontsize=9, color="#4b5563")
    fig.savefig(destination / "same_streets_old_vs_current.png", dpi=180, facecolor="white")
    fig.savefig(destination / "same_streets_old_vs_current.pdf", facecolor="white")
    plt.close(fig)
    metadata = {
        "method": "CPU plotting and metrics from existing saved FP32 model outputs; no new inference or training",
        "inputs_exactly_equal": True, "displayed_indices": [0, 1, 2],
        "old": {"variant": "multiscale_residual_ista", "checkpoint_step": 6000,
                "checkpoint": str(old_root / "runs/multiscale_residual_ista/checkpoint_best.pt"),
                "arrays": str(old_source), "checkpoint_verification": verification,
                "gene_cap": None, "expression_grids": [16, 32, 64], "expression_type": "float"},
        "current": {"variant": "learned_s42", "checkpoint_step": 7500,
                    "checkpoint": str(root / "runs/learned_s42/checkpoint_best.pt"),
                    "arrays": str(current_source), "source_metadata": str(current_reports / "metadata.json"),
                    "gene_cap": 64, "expression_grids": [8, 16, 32], "expression_type": "uint8, range [0,4]"},
        "city128": {"old": {"mae": float(old_all_mae.mean()), "psnr_mean_per_image": float(old_all_psnr.mean())},
                    "current_cap64": {k: rate64[k] for k in ("l1", "psnr_mean_per_image", "lpips", "images", "mean_active_genes")}},
        "displayed_measurements": measurements,
        "interpretation": "Observed quality regression across different architectures and code capacities; not an equal-budget controlled ablation.",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), "city128": metadata["city128"], "displayed_measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
