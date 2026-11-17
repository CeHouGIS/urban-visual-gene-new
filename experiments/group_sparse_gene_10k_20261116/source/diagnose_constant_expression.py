"""Decode constant-expression interventions on one leased GPU; no training."""
import json
import os
from pathlib import Path

import numpy as np
from common import load_config
from reference.supervisor import acquire


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiment.json")
    destination = root / "reports/constant_expression_cap32"
    destination.mkdir(exist_ok=True)
    leases = acquire(config, 1)
    if not leases:
        (destination / "status.json").write_text(json.dumps({"status": "waiting_for_gpu"}) + "\n")
        print("No idle GPU lease available; no CPU model execution.")
        return
    gpu, lease = leases[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    try:
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from model import create_model

        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        pair = json.loads((root / "reports/cap32_shared_gene_pair_chattogram_kolkata.json").read_text())
        saved = torch.load(root / "runs/learned_s42/checkpoint_best.pt", map_location="cpu", weights_only=False)
        model = create_model(config, "learned_s42").eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        model.set_step(saved["step"])
        model.rate_index = 2
        assert saved["step"] == pair["checkpoint_step"] == 7500
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved["model"][key]), key
        with np.load(pair["source"]) as data:
            codes = {s: torch.from_numpy(data[f"cap32_{s}_uint8"].astype(np.float32) * (4. / 255.)).cuda()
                     for s in model.scales}
        constant, rounded, zero = {}, {}, {}
        for scale, values in codes.items():
            ids = pair["scales"][scale]["shared_gene_ids"]
            constant[scale] = torch.zeros_like(values)
            constant[scale][:, ids] = 1.
            rounded[scale] = constant[scale] * (64 * 4. / 255.)
            zero[scale] = torch.zeros_like(values)
            assert torch.equal(constant[scale][0], constant[scale][1])
        one_replaced = {s: value.clone() for s, value in codes.items()}
        one_replaced["fine"][:, 125] = 1.
        one_only = {s: value.clone() for s, value in zero.items()}
        one_only["fine"][:, 125] = 1.
        variants = {"original_codes": codes, "all_32_exact_ones": constant,
                    "all_32_quantized_ones": rounded, "all_zero": zero,
                    "fine125_one_keep_others": one_replaced, "fine125_one_only": one_only}
        arrays, measurements = {}, {}
        original_directory = root / "reports/cap32_shared_gene_pair_chattogram_kolkata"
        originals = np.stack([np.asarray(Image.open(original_directory / f"{city}_original.png")).astype(np.float32) / 255
                              for city in ["chattogram", "kolkata"]])
        target = torch.from_numpy(originals).permute(0, 3, 1, 2).cuda()
        with torch.inference_mode():
            for name, activation in variants.items():
                output = model.decoder(model.synthesize_codes(activation)).float()
                assert torch.isfinite(output).all()
                arrays[name] = output.cpu().numpy()
                measurements[name] = dict(
                    image_mae=(output - target).abs().mean((1, 2, 3)).cpu().tolist(),
                    output_pair_max_abs_difference=float((output[0] - output[1]).abs().max()),
                    rgb_min=float(output.min()), rgb_max=float(output.max()),
                    spatial_std_per_channel=output.std((2, 3)).cpu().tolist())
            assert measurements["all_32_exact_ones"]["output_pair_max_abs_difference"] == 0.
            assert measurements["all_32_quantized_ones"]["output_pair_max_abs_difference"] == 0.
        arrays["originals"] = originals.transpose(0, 3, 1, 2)
        for scale, values in constant.items():
            arrays[f"exact_one_codes_{scale}"] = values.cpu().numpy()
        np.savez_compressed(destination / "constant_expression_arrays.npz", **arrays)

        def rgb(name, index):
            return arrays[name][index].transpose(1, 2, 0).clip(0, 1)

        def clean(ax):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.7))
        for row, city in enumerate(["Chattogram", "Kolkata"]):
            for col, name in enumerate(["originals", "original_codes", "all_32_exact_ones"]):
                axes[row, col].imshow(rgb(name, row)); clean(axes[row, col])
                if row == 0:
                    axes[row, col].set_title(["Original photo", "Original spatial expressions", "All 32 expressions = 1"][col])
            axes[row, 0].set_ylabel(city, fontsize=13)
        fig.suptitle("Replacing spatial expressions with ones", fontsize=18, weight="bold", y=.98)
        fig.text(.5, .025, "Same 32 gene IDs and fixed weights. The right-column outputs are exactly identical.", ha="center", fontsize=11)
        fig.tight_layout(rect=[0, .055, 1, .94], h_pad=2.4)
        fig.savefig(destination / "all_ones_vs_original_codes.png", dpi=180, facecolor="white")
        fig.savefig(destination / "all_ones_vs_original_codes.pdf", facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1))
        for ax, name, label in zip(axes,
                ["original_codes", "fine125_one_keep_others", "fine125_one_only", "all_zero"],
                ["Original expressions (A)", "Only fine:125 changed to 1", "Only fine:125 = 1; rest zero", "All expressions = 0"]):
            ax.imshow(rgb(name, 0)); clean(ax); ax.set_title(label, fontsize=10)
        fig.suptitle("One gene at constant expression: fine:125", fontsize=16, y=.97)
        fig.tight_layout(rect=[0, 0, 1, .9])
        fig.savefig(destination / "single_gene_constant_controls.png", dpi=180, facecolor="white")
        plt.close(fig)
        for name in variants:
            Image.fromarray(np.rint(rgb(name, 0) * 255).astype(np.uint8)).save(destination / f"{name}.png")
        measurements["exact_vs_quantized_one_output_mae"] = float(np.abs(arrays["all_32_exact_ones"] - arrays["all_32_quantized_ones"]).mean())
        metadata = dict(status="complete", variant="learned_s42", checkpoint_step=7500, cap=32,
            physical_gpu=gpu, training=False, source_codes=pair["source"], shared_gene_ids=pair["scales"],
            interventions="Exact 1 is a direct pre-decoder counterfactual; also decode Q8(1)=256/255 for legal 8-bit expressions. Unselected genes remain zero.",
            measurements=measurements)
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        (destination / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
        print(json.dumps({"destination": str(destination), "measurements": measurements}, indent=2))
    finally:
        lease.close()


if __name__ == "__main__":
    main()
