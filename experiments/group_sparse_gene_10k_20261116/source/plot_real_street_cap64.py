"""Display saved, checkpoint-verified cap64 street reconstructions and errors."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/scale_gene_views_20261117_042541"
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["variant"] == "learned_s42"
    assert metadata["checkpoint_step"] == 7500 and metadata["primary_gene_cap"] == 64
    assert metadata["image_indices"] == [0, 1, 2]
    with np.load(source / "scale_gene_arrays.npz") as data:
        originals = data["original"].copy()
        outputs = data["full_reconstruction"].copy()
    assert originals.shape == outputs.shape == (3, 3, 256, 256)
    assert np.isfinite(outputs).all()
    error = np.abs(outputs - originals).mean(axis=1)
    destination = root / "reports/real_street_input_cap64"
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / "reconstruction_arrays.npz", originals=originals,
                        reconstructions=outputs, mean_abs_rgb_error=error)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.3))
    fig.subplots_adjust(left=.12, right=.87, bottom=.07, top=.90, hspace=.23, wspace=.07)
    measurements = []
    for row, sample in enumerate(metadata["images"]):
        x = originals[row].transpose(1, 2, 0)
        y = outputs[row].transpose(1, 2, 0)
        axes[row, 0].imshow(x.clip(0, 1), interpolation="nearest")
        axes[row, 1].imshow(y.clip(0, 1), interpolation="nearest")
        heatmap = axes[row, 2].imshow(error[row], cmap="inferno", vmin=0, vmax=.25, interpolation="nearest")
        axes[row, 0].set_ylabel(sample["city"] + "\n" + sample["country"], fontsize=11)
        mae = float(error[row].mean())
        mse = float(np.square(outputs[row] - originals[row]).mean())
        axes[row, 1].set_xlabel(f"MAE {mae:.5f}", fontsize=11)
        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(["Input street photo", "Output: 64 genes", "Absolute RGB error"][col], fontsize=12)
        for suffix, values in [("input", x), ("output", y)]:
            Image.fromarray(np.rint(values.clip(0, 1) * 255).astype(np.uint8)).save(
                destination / f'{sample["city"].lower()}_{suffix}.png')
        measurements.append(dict(city=sample["city"], country=sample["country"], image_id=sample["id"],
                                  mae=mae, psnr=float(-10 * np.log10(max(mse, 1e-12)))))
    colorbar = fig.colorbar(heatmap, cax=fig.add_axes([.90, .24, .018, .50]))
    colorbar.set_label("Mean absolute RGB error per pixel", fontsize=10)
    fig.suptitle("Real street inputs through the complete 64-gene AE", fontsize=16, weight="bold", y=.975)
    fig.text(.12, .935, "Same learned_s42 checkpoint 7500 as the white-image and noise-input probes", fontsize=10, color="#4b5563")
    fig.text(.12, .025, "RGB range [0,1]. Error color scale: 0-0.25 (larger values saturate). Lille input includes privacy blur.", fontsize=8, color="#4b5563")
    fig.savefig(destination / "real_street_input_output_error.png", dpi=190, facecolor="white")
    fig.savefig(destination / "real_street_input_output_error.pdf", facecolor="white")
    plt.close(fig)
    result = dict(variant="learned_s42", checkpoint_step=7500, gene_cap=64,
                  source=str(source / "scale_gene_arrays.npz"), image_indices=[0, 1, 2],
                  source_note="Existing full-model FP32 inference, strict checkpoint verification in source/visualize_scale_genes.py; no new model execution.",
                  error_definition="Per-pixel mean absolute RGB reconstruction error against input, fixed color scale [0,.25]",
                  measurements=measurements)
    (destination / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(destination=str(destination), measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
