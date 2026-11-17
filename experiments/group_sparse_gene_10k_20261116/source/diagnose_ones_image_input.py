"""Run a white or fixed-seed random RGB image through the complete trained AE."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from common import load_config
from reference.supervisor import acquire


def main(cap=32, input_kind='ones', seed=42):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiment.json")
    directory = f'all_ones_rgb_input_cap{cap}' if input_kind == 'ones' else f'random_rgb_input_seed{seed}_cap{cap}'
    destination = root / 'reports' / directory
    destination.mkdir(exist_ok=True)
    leases = acquire(config, 1)
    if not leases:
        (destination / "status.json").write_text('{"status": "waiting_for_gpu"}\n')
        print("Waiting for an idle leased GPU; no CPU inference.")
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
        checkpoint = root / "runs/learned_s42/checkpoint_best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = create_model(config, "learned_s42").eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        model.set_step(saved["step"])
        model.rate_index = config['model']['budgets'].index(cap)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved["model"][key]), key
        with torch.inference_mode():
            if input_kind == 'ones':
                image = torch.ones(1, 3, 256, 256, device="cuda", dtype=torch.float32)
            else:
                rng = np.random.default_rng(seed)
                pixels = rng.integers(0, 256, size=(1, 3, 256, 256), dtype=np.uint8)
                image = torch.from_numpy(pixels.astype(np.float32) / 255.).cuda()
            result = model(image)
            output = result["reconstruction"].float()
            assert output.shape == image.shape and torch.isfinite(output).all()
            count, byte_count = model.actual_complexity(result)
            assert int(count[0]) <= cap
            mae = float((output - image).abs().mean())
            mse = float((output - image).square().mean())
            stats = dict(input_shape=list(image.shape), input_min=float(image.min()), input_max=float(image.max()),
                         input_mean_rgb=image.mean((0, 2, 3)).cpu().tolist(),
                         input_spatial_std_rgb=image.std((0, 2, 3)).cpu().tolist(),
                         active_genes=int(count[0]), encoded_bytes=int(byte_count[0]),
                         mae=mae, mse=mse, psnr=-10 * float(np.log10(max(mse, 1e-12))),
                         output_min=float(output.min()), output_max=float(output.max()),
                         mean_rgb=output.mean((0, 2, 3)).cpu().tolist(),
                         spatial_std_rgb=output.std((0, 2, 3)).cpu().tolist(),
                         expressions={})
            codes = {}
            for scale, activation in result["activations"].items():
                used = (activation[0] != 0).any((1, 2)).nonzero().flatten()
                assert len(used) <= model.quota(scale)
                stats["expressions"][scale] = dict(gene_ids=used.cpu().tolist(),
                    shape=list(activation.shape), min=float(activation[:, used].min()),
                    max=float(activation[:, used].max()), mean=float(activation[:, used].mean()))
                codes[scale] = activation.cpu().numpy()
            x = image[0].cpu().permute(1, 2, 0).numpy()
            y = output[0].cpu().permute(1, 2, 0).numpy()
        np.save(destination / "input_rgb_float32.npy", x)
        np.save(destination / "output_rgb_float32.npy", y)
        np.savez_compressed(destination / "encoded_expression_maps.npz", **codes)
        input_filename = 'input_white.png' if input_kind == 'ones' else f'input_noise_seed{seed}.png'
        for name, values in [(input_filename, x), (f"output_cap{cap}.png", y)]:
            Image.fromarray(np.rint(values.clip(0, 1) * 255).astype(np.uint8)).save(destination / name)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.9))
        input_title = 'Input: RGB values all = 1' if input_kind == 'ones' else f'Input: random RGB noise (seed {seed})'
        for ax, values, title in zip(axes, [x, y], [input_title, f"Output: full AE, cap {cap}"]):
            ax.imshow(values.clip(0, 1), interpolation="nearest")
            ax.set_title(title, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor("#9ca3af")
        title = 'All-one image' if input_kind == 'ones' else 'Random RGB noise'
        fig.suptitle(f'{title} through Encoder -> Gene synthesis -> Decoder', fontsize=14, y=.98)
        fig.text(.5, .04, f"256 x 256 RGB, range [0,1]  |  Actual active genes: {int(count[0])}  |  MAE: {mae:.5f}",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=[.015, .08, .985, .94])
        figure_name = 'ones_image_input_output' if input_kind == 'ones' else 'random_image_input_output'
        fig.savefig(destination / f'{figure_name}.png', dpi=200, facecolor="white")
        fig.savefig(destination / f'{figure_name}.pdf', facecolor="white")
        plt.close(fig)
        description = ('Input image is all-one float32 RGB in [0,1].' if input_kind == 'ones' else
                       f'Input pixels are independent uniform uint8 RGB noise (0..255), NumPy default_rng seed {seed}, divided by 255.')
        metadata = dict(status="complete", variant="learned_s42", checkpoint=str(checkpoint),
                        checkpoint_step=saved["step"], gene_cap=cap, physical_gpu=gpu,
                        input_kind=input_kind, noise_seed=seed if input_kind == 'random' else None,
                        intervention=description + ' Complete model forward; no expression edits.',
                        statistics=stats)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        prior_directory = root / 'reports/all_ones_rgb_input_cap32'
        if input_kind == 'ones' and cap == 64 and (prior_directory / 'output_rgb_float32.npy').exists():
            prior_metadata = json.loads((prior_directory / 'metadata.json').read_text())
            assert prior_metadata['checkpoint_step'] == saved['step']
            assert prior_metadata['variant'] == 'learned_s42'
            prior = np.load(prior_directory / 'output_rgb_float32.npy')
            assert prior.shape == y.shape
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
            labels = ['Input: all-one RGB',
                      f"32 genes | MAE {prior_metadata['statistics']['mae']:.5f}",
                      f'64 genes | MAE {mae:.5f}']
            for ax, values, title in zip(axes, [x, prior, y], labels):
                ax.imshow(values.clip(0, 1), interpolation='nearest')
                ax.set_title(title, fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#9ca3af')
            fig.suptitle('All-one image input: 32 vs. 64 genes', fontsize=16, y=.98)
            fig.tight_layout(rect=[.01, .025, .99, .91])
            fig.savefig(destination / 'ones_image_cap32_vs_cap64.png', dpi=200, facecolor='white')
            fig.savefig(destination / 'ones_image_cap32_vs_cap64.pdf', facecolor='white')
            plt.close(fig)
        (destination / "status.json").write_text('{"status": "complete"}\n')
        print(json.dumps(dict(destination=str(destination), statistics=stats), indent=2))
    finally:
        lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene-cap', type=int, choices=[8, 16, 32, 64], default=32)
    parser.add_argument('--input-kind', choices=['ones', 'random'], default='ones')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.gene_cap, args.input_kind, args.seed)
