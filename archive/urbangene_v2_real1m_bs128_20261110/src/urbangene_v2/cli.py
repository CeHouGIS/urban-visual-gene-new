import argparse
import json

from .common import OUTPUT_ROOT, load_config, write_json
from .data import prepare_synthetic
from .extract import extract_manifest
from .train import evaluate_discovery, train_discovery


def main():
    parser = argparse.ArgumentParser(prog="urbangene_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", default=str(OUTPUT_ROOT / "baseline_audit.json"))
    extract = sub.add_parser("extract-features")
    extract.add_argument("--config", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare-synthetic")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    train = sub.add_parser("train-discovery")
    train.add_argument("--config", required=True)
    evaluate = sub.add_parser("eval-discovery")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--grid-hw", nargs=2, type=int)
    evaluate.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "audit":
        audit_result = {
            "v1_1": {"status": "active queue; read-only audit",
                     "queue": "/host/mnt/nas/urban-visual-gene/urbangene_v11_20261108/v1_grct/formal_v11_20261108_activeguard",
                     "pid": 399150, "reported_completed_runs": 3,
                     "reported_total_runs": 14, "source": "/root/code/urban-visual-gene/urbangene_v11/EXECUTION_STATUS.md"},
            "v1_2_fixed_grid_rvq": {"status": "not found in repository or project NAS outputs; no claim of implementation/run"},
            "dino_v3": {"full_resolution_feature_cache": "/host/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000/dict_<city>_p784.f16.npy",
                        "feature_shape_per_city": [9408000, 1024], "dtype": "float16",
                        "checkpoint": "not yet located; see model_search.json"},
            "isolation": {"v2_source": "NAS-backed urbangene_v2_20261109/source",
                          "v2_outputs": str(OUTPUT_ROOT), "legacy_outputs_write_allowed": False},
        }
        write_json(args.output, audit_result)
        print(args.output)
    elif args.command == "extract-features":
        config = load_config(args.config)
        metadata = extract_manifest(args.manifest, args.output, config["backbone"],
                                    config.get("batch_size", 1), config.get("workers", 0),
                                    config.get("device", "cpu"))
        print(json.dumps(metadata, indent=2))
    elif args.command == "prepare-synthetic":
        config = load_config(args.config)
        prepare_synthetic(config, args.output)
        print(args.output)
    elif args.command == "train-discovery":
        output = train_discovery(load_config(args.config))
        print(output)
    elif args.command == "eval-discovery":
        summary = evaluate_discovery(args.checkpoint, args.features, args.annotations,
                                     args.grid_hw, args.device, args.output)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
