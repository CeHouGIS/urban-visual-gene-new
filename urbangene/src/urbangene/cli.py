import argparse
from pathlib import Path

from .utils import load_config


def main():
    parser = argparse.ArgumentParser(description="UrbanGene V0 reproducible training")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["generate", "train-ae", "train", "plan", "train-probe"]:
        p = sub.add_parser(name); p.add_argument("--config", required=True)
        if name != "generate":
            p.add_argument("--output", required=True)
        if name in ["train-ae", "train", "train-probe"]:
            p.add_argument("--device", default="cpu")
        if name in ["train-ae", "train"]:
            p.add_argument("--resume")
        if name == "train":
            p.add_argument("--ae", required=True)
            p.add_argument("--method", choices=["urbangene", "b0", "b1", "b2"], default="urbangene")
    p = sub.add_parser("evaluate"); p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--max-samples", type=int); p.add_argument("--probe")
    p = sub.add_parser("queue"); p.add_argument("--plan", required=True)
    p.add_argument("--device"); p.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.checkpoint, args.split, args.device, args.max_samples, args.probe)
        return
    if args.command == "queue":
        from .schedule import execute_plan
        execute_plan(args.plan, not args.no_wait, args.device)
        return
    cfg = load_config(args.config)
    if args.command == "generate":
        from .data.synthetic import SyntheticSceneGenerator, generate_manifests
        from .visualize import scene_grid
        specs = generate_manifests(cfg, cfg["data"]["root"])
        for split, values in specs.items():
            scene_grid(Path(cfg["data"]["root"]) / f"{split}_preview.png", SyntheticSceneGenerator(), values)
        print({k: len(v) for k, v in specs.items()})
    elif args.command == "train-ae":
        from .train import train_ae
        train_ae(cfg, args.output, args.device, args.resume)
    elif args.command == "train":
        from .train import train_gene
        train_gene(cfg, args.ae, args.output, args.method, args.device, args.resume)
    elif args.command == "plan":
        from .schedule import build_plan
        plan = build_plan(cfg, args.output)
        print(f"Prepared {len(plan['runs'])} serial Gene runs with validation gates")
    elif args.command == "train-probe":
        from .probe import train_probe
        train_probe(cfg, args.output, args.device)


if __name__ == "__main__":
    main()
