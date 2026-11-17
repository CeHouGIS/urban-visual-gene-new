"""Compare saved cap32 expression codes without running a neural network."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plain(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def tile_atom(atom):
    channels, height, width = atom.shape
    assert channels == 64 and height == width
    tiled = np.full((8 * (height + 1) - 1, 8 * (width + 1) - 1), np.nan)
    for c in range(channels):
        y, x = divmod(c, 8)
        tiled[y * (height + 1):y * (height + 1) + height,
              x * (width + 1):x * (width + 1) + width] = atom[c]
    return tiled


def main():
    root = Path(__file__).resolve().parents[1]
    pair_path = root / "reports/cap32_shared_gene_pair_chattogram_kolkata.json"
    pair = json.loads(pair_path.read_text())
    code_path = Path(pair["source"])
    preview_metadata = json.loads(code_path.with_name("city_confirmation.json").read_text())
    atom_directory = root / "reports/scale_gene_views_20261117_042541"
    atom_metadata = json.loads((atom_directory / "metadata.json").read_text())
    assert atom_metadata["variant"] == pair["variant"] == "learned_s42"
    assert atom_metadata["checkpoint_step"] == preview_metadata["step"] == pair["checkpoint_step"] == 7500
    output = root / "reports/cap32_gene_expression_comparison"
    output.mkdir(exist_ok=True)
    scales = ["coarse", "medium", "fine"]
    selected = {"coarse": 52, "medium": 93, "fine": 125}
    expressions, weights, records = {}, {}, []
    arrays = {}
    with np.load(code_path) as data, np.load(atom_directory / "scale_gene_arrays.npz") as atoms:
        for scale in scales:
            quantized = data[f"cap32_{scale}_uint8"]
            assert quantized.shape[0] == 2 and quantized.dtype == np.uint8
            used = [np.flatnonzero(np.any(q != 0, axis=(1, 2))) for q in quantized]
            assert np.array_equal(used[0], used[1])
            assert used[0].tolist() == pair["scales"][scale]["shared_gene_ids"]
            expression = quantized.astype(np.float32) * (4. / 255.)
            expressions[scale] = {int(g): expression[:, g] for g in used[0]}
            assert selected[scale] == atom_metadata[scale]["gene"]
            weights[scale] = atoms[f"{scale}_atom"].copy()
            arrays[f"{scale}_gene_ids"] = used[0]
            arrays[f"{scale}_expression_uint8"] = quantized[:, used[0]]
            arrays[f"{scale}_example_atom"] = weights[scale]
            for g in used[0]:
                a, b = expressions[scale][int(g)]
                delta = b - a
                records.append(dict(scale=scale, gene=int(g), height=a.shape[0], width=a.shape[1],
                    mean_expression_A=float(a.mean()), mean_expression_B=float(b.mean()),
                    mean_abs_expression_difference=float(np.abs(delta).mean()),
                    max_abs_expression_difference=float(np.abs(delta).max()),
                    changed_cell_fraction=float(np.mean(a != b))))
    assert len(records) == 32 and all(r["changed_cell_fraction"] > 0 for r in records)
    arrays["image_indices"] = np.array([0, 63])
    np.savez_compressed(output / "expression_comparison_arrays.npz", **arrays)
    with (output / "expression_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    expression_norm = Normalize(0, 4)
    # One common signed scale for every displayed difference; no clipping.
    delta_max = float(np.ceil(max(r["max_abs_expression_difference"] for r in records) * 10) / 10)
    delta_norm = Normalize(-delta_max, delta_max)
    weight_max = max(float(np.abs(w).max()) for w in weights.values())
    weight_norm = Normalize(-weight_max, weight_max)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    image_directory = root / "reports/cap32_shared_gene_pair_chattogram_kolkata"
    originals = [np.asarray(Image.open(image_directory / f"{city}_original.png"))
                 for city in ["chattogram", "kolkata"]]

    summary = plt.figure(figsize=(13.8, 12.2))
    gs = summary.add_gridspec(4, 4, height_ratios=[.7, 1, 1, 1], width_ratios=[.85, 1, 1, 1],
                             left=.075, right=.97, bottom=.11, top=.90, hspace=.37, wspace=.22)
    summary.suptitle("Same gene weights; different spatial expressions", fontsize=19, weight="bold", y=.98)
    summary.text(.075, .935, "Cap 32  |  learned_s42, checkpoint 7500  |  three previously illustrated genes",
                 fontsize=11, color="#4b5563")
    ax = summary.add_subplot(gs[0, 0]); plain(ax)
    ax.text(.05, .65, "Gene weights D\nShared by both images\n64 feature channels\nColors are not RGB", va="center", fontsize=11)
    for j, (picture, label) in enumerate(zip(originals, ["A: Chattogram", "B: Kolkata"]), 1):
        ax = summary.add_subplot(gs[0, j]); ax.imshow(picture); plain(ax); ax.set_title(label, fontsize=12)
    ax = summary.add_subplot(gs[0, 3]); plain(ax)
    ax.text(.02, .65, "Expression difference B - A\nRed: stronger in Kolkata\nBlue: stronger in Chattogram\nWhite: similar expression", va="center", fontsize=11)
    for i, scale in enumerate(scales, 1):
        gene = selected[scale]; a, b = expressions[scale][gene]
        ax = summary.add_subplot(gs[i, 0]); plain(ax)
        ax.imshow(tile_atom(weights[scale]), cmap="RdBu_r", norm=weight_norm, interpolation="nearest")
        ax.set_ylabel(f"{scale}:{gene}\nD shape {tuple(weights[scale].shape)}", fontsize=11)
        ax.set_title("Shared dictionary weights", fontsize=10)
        for j, values in enumerate([a, b], 1):
            ax = summary.add_subplot(gs[i, j]); plain(ax)
            ax.imshow(values, cmap="magma", norm=expression_norm, interpolation="nearest")
            ax.set_title(f"Expression {'A' if j == 1 else 'B'} | {a.shape[0]} x {a.shape[1]}", fontsize=10)
            ax.set_xlabel(f"Mean expression {values.mean():.3f}", fontsize=9)
        ax = summary.add_subplot(gs[i, 3]); plain(ax)
        ax.imshow(b - a, cmap="RdBu_r", norm=delta_norm, interpolation="nearest")
        ax.set_title("B - A", fontsize=10)
        ax.set_xlabel(f"Mean |difference| {np.abs(b-a).mean():.3f}", fontsize=9)
    for rect, norm, cmap, label in [([.10, .055, .15, .012], weight_norm, "RdBu_r", "Dictionary weight (shared color scale)"),
                                  ([.36, .055, .25, .012], expression_norm, "magma", "Expression amplitude (fixed 0 to 4)"),
                                  ([.73, .055, .20, .012], delta_norm, "RdBu_r", "Signed expression difference (B - A)")]:
        bar = summary.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=summary.add_axes(rect), orientation="horizontal")
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=8)
    summary.savefig(output / "three_gene_expression_comparison.png", dpi=170, facecolor="white")
    summary.savefig(output / "three_gene_expression_comparison.svg", facecolor="white")

    atlas_names = []
    with PdfPages(output / "all_32_genes_comparison.pdf") as pdf:
        pdf.savefig(summary)
        plt.close(summary)
        for scale in scales:
            ids = sorted(expressions[scale])
            for start in range(0, len(ids), 8):
                genes = ids[start:start + 8]
                fig = plt.figure(figsize=(14.8, 12.8))
                grid = fig.add_gridspec(4, 2, left=.045, right=.975, bottom=.09, top=.85, hspace=.42, wspace=.18)
                fig.suptitle(f"{scale.capitalize()} genes: same IDs, different expressions", fontsize=19, weight="bold", y=.98)
                fig.text(.045, .934, "A: Chattogram   |   B: Kolkata   |   fixed expression scale 0-4; common signed difference scale", fontsize=11)
                for n, gene in enumerate(genes):
                    sub = grid[n // 2, n % 2].subgridspec(1, 3, wspace=.08)
                    a, b = expressions[scale][gene]
                    axes = [fig.add_subplot(sub[0, j]) for j in range(3)]
                    for j, (ax, values) in enumerate(zip(axes, [a, b, b-a])):
                        ax.imshow(values, cmap="magma" if j < 2 else "RdBu_r",
                                  norm=expression_norm if j < 2 else delta_norm, interpolation="nearest")
                        plain(ax); ax.set_title(["A", "B", "B - A"][j], fontsize=10, pad=3)
                    bounds = axes[0].get_position()
                    fig.text(bounds.x0, bounds.y1 + .035, f"{scale}:{gene}  |  mean |difference| = {np.abs(b-a).mean():.3f}", fontsize=11, weight="bold")
                for rect, norm, cmap, label in [([.16, .035, .27, .012], expression_norm, "magma", "Expression amplitude"),
                                               ([.60, .035, .27, .012], delta_norm, "RdBu_r", "Signed difference B - A")]:
                    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(rect), orientation="horizontal")
                    bar.set_label(label, fontsize=9)
                name = f"{scale}_genes_{start // 8 + 1}.png"
                fig.savefig(output / name, dpi=160, facecolor="white")
                pdf.savefig(fig); plt.close(fig); atlas_names.append(name)

    metadata = dict(variant=pair["variant"], checkpoint_step=7500, cap=32, images=pair["images"],
                    sources=[str(code_path), str(atom_directory / "scale_gene_arrays.npz")],
                    selected_examples=selected, selection="Previously illustrated coarse52 / medium93 / fine125; not selected for maximum difference",
                    total_shared_genes=32, genes_with_different_expression_maps=32,
                    expression_color_range=[0, 4], difference_definition="Kolkata expression minus Chattogram expression at the same normalized grid coordinates",
                    difference_color_range=[-delta_max, delta_max],
                    interpretation="Expression maps are quantized latent coefficients, not RGB colors, object masks, or deletion contributions. Coordinates are within each photo, not matched physical locations.",
                    atlas_files=atlas_names, measurements=records)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text("# 同一组32个基因在两张街景中的表达差异\n\n"
        "来源：learned_s42，checkpoint 7500，cap32；直接读取已保存的uint8表达码，无新训练或神经网络推理。\n\n"
        "A为Chattogram，B为Kolkata。每个gene的字典权重D在两张图间相同；逐图表达A不同。"
        "主图展示此前使用过的coarse52、medium93、fine125；PDF首张为主图，后四张覆盖全部32个gene。\n\n"
        "表达图统一色阶[0,4]。差值为B-A，所有图使用同一对称色阶且未截断。红色表示B更强、蓝色表示A更强。"
        "比较的是各图片内部相同归一化坐标，不代表同一个地理位置。权重颜色不是RGB，表达不是分割或删除贡献。\n\n"
        "CSV包含每个gene的均值、表达差异和变化网格比例；NPZ保存实际uint8表达码和三个例子的字典权重。\n")
    print(json.dumps(dict(output=str(output), shared_genes=32, changed_expressions=32, atlas_files=atlas_names), indent=2))


if __name__ == "__main__":
    main()
