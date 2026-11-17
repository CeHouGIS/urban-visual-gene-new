import argparse
import copy
from pathlib import Path

import yaml

from .config import PROJECT, load_config
from .evaluation import evaluate_recombination, evaluate_reconstruction
from .training import train


def _generate(cfg):
    from urbangene.data.synthetic import generate_manifests
    from urbangene.visualize import scene_grid
    from urbangene.data.synthetic import SyntheticSceneGenerator

    root = Path(cfg["data"]["root"])
    if not (root / "audit.json").exists():
        specs = generate_manifests(cfg, root)
    else:
        import json
        specs = {p.stem: [json.loads(line) for line in p.read_text().splitlines()]
                 for p in root.glob("*.jsonl")}
    for split, values in specs.items():
        preview = root / f"{split}_preview.png"
        if not preview.exists():
            scene_grid(preview, SyntheticSceneGenerator(), values)
    return root


def smoke(cfg_path, device="cpu", output=None):
    from urbangene.train import train_ae
    from urbangene.utils import save_json

    cfg = load_config(cfg_path)
    root = Path(cfg["data"]["root"])
    _generate(cfg)
    base = Path(output) if output is not None else PROJECT / "outputs/v1_grct/smoke"
    ae_dir = base / "ae"
    runs = {name: base / name for name in ("r0_fixed", "r2_residual_cnn", "r3_grct")}
    save_json(base / "plan.json", dict(kind="cpu_smoke", device=device, config=str(Path(cfg_path).resolve()),
        methods={name: str(path) for name, path in runs.items()}, shared_ae=str(ae_dir)))
    ae_path = ae_dir / "best.pt"
    if not ae_path.exists():
        ae_path = train_ae(cfg, ae_dir, device, ae_dir / "last.pt" if (ae_dir / "last.pt").exists() else None)
    results = {}
    for name, composer in (("r0_fixed", "fixed_average"),
                           ("r2_residual_cnn", "residual_cnn"), ("r3_grct", "grct")):
        method_cfg = copy.deepcopy(cfg); method_cfg["model"]["composer"]["type"] = composer
        checkpoint = runs[name] / "best.pt"
        if not checkpoint.exists():
            resume = runs[name] / "last.pt" if (runs[name] / "last.pt").exists() else None
            checkpoint = train(method_cfg, ae_path, runs[name], device, resume)
        results[name] = str(checkpoint)
    for name, checkpoint in results.items():
        recon_metrics = runs[name] / "eval/test_seen/metrics.json"
        pair_metrics = runs[name] / "eval/recombination/summary.json"
        if not recon_metrics.exists():
            evaluate_reconstruction(checkpoint, "test_seen", runs[name] / "eval/test_seen", device)
        if not pair_metrics.exists():
            evaluate_recombination(checkpoint, runs[name] / "eval/recombination", device,
                                   cfg["evaluation"]["recombination_samples"], cfg["seed"])
    save_json(base / "status.json", dict(state="finished", ae=str(ae_path), checkpoints=results,
        note="CPU diagnostic only; smoke data and reduced architecture are not formal results"))
    return results


def main():
    parser = argparse.ArgumentParser(description="UrbanGene V1.1 GRCT experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "train-ae", "train", "smoke"):
        p = sub.add_parser(name); p.add_argument("--config", required=True)
        if name in ("train-ae", "train"):
            p.add_argument("--device", default="cpu")
        if name in ("train-ae", "train"):
            p.add_argument("--output", required=True); p.add_argument("--resume")
        if name == "train":
            p.add_argument("--ae", required=True)
        if name == "smoke":
            p.add_argument("--device", default="cpu"); p.add_argument("--output")
    p = sub.add_parser("evaluate"); p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--max-samples", type=int)
    p = sub.add_parser("recombine"); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--count", type=int); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("plan"); p.add_argument("--output", required=True)
    p = sub.add_parser("queue"); p.add_argument("--plan", required=True)
    p.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        print(evaluate_reconstruction(args.checkpoint, args.split, args.output, args.device, args.max_samples))
    elif args.command == "recombine":
        evaluate_recombination(args.checkpoint, args.output, args.device, args.count, args.seed)
    elif args.command == "queue":
        from .schedule import execute_plan
        execute_plan(args.plan, wait_for_gpu=not args.no_wait)
    elif args.command == "plan":
        from .schedule import build_plan
        print(build_plan(args.output))
    else:
        cfg = load_config(args.config)
        if args.command == "generate":
            print(_generate(cfg))
        elif args.command == "train-ae":
            from urbangene.train import train_ae
            train_ae(cfg, args.output, args.device, args.resume)
        elif args.command == "train":
            train(cfg, args.ae, args.output, args.device, args.resume)
        elif args.command == "smoke":
            smoke(args.config, args.device, args.output)


if __name__ == "__main__":
    main()
