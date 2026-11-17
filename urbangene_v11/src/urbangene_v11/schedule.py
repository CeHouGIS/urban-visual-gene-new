import fcntl
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from urbangene.data.dataset import SceneDataset
from urbangene.metrics import image_metrics
from urbangene.train import load_ae, make_loader, train_ae
from urbangene.utils import save_json

from .config import PROJECT, load_config
from .evaluation import evaluate_recombination, evaluate_reconstruction
from .training import train


V0 = PROJECT.parent / "urbangene"
METHODS = [
    ("r0_fixed", "v1_r0_fixed.yaml", 42),
    ("r2_residual_cnn", "v1_r2_residual_cnn.yaml", 42),
    ("r3_grct", "v1_r3_grct.yaml", 42),
    ("r1_alpha", "v1_r1_alpha.yaml", 42),
    ("r3_no_relation", "v1_r3_no_relation.yaml", 42),
    ("r3_no_spatial", "v1_r3_no_spatial.yaml", 42),
    ("r0_parameter_matched", "v1_r0_parameter_matched.yaml", 42),
    ("r2_parameter_matched", "v1_r2_parameter_matched.yaml", 42),
    ("r0_fixed", "v1_r0_fixed.yaml", 29),
    ("r2_residual_cnn", "v1_r2_residual_cnn.yaml", 29),
    ("r3_grct", "v1_r3_grct.yaml", 29),
    ("r0_fixed", "v1_r0_fixed.yaml", 43),
    ("r2_residual_cnn", "v1_r2_residual_cnn.yaml", 43),
    ("r3_grct", "v1_r3_grct.yaml", 43),
]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_manifest():
    roots = [PROJECT / "src/urbangene_v11", V0 / "src/urbangene"]
    paths = sorted(path for root in roots for path in root.rglob("*.py"))
    return {str(path.relative_to(PROJECT.parent)): _sha(path) for path in paths}


def build_plan(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    snapshot.mkdir()
    source_hashes = _source_manifest()
    shutil.copytree(PROJECT / "src/urbangene_v11", snapshot / "urbangene_v11")
    shutil.copytree(V0 / "src/urbangene", snapshot / "urbangene_v0")
    shutil.copytree(V0 / "configs", snapshot / "v0_configs")
    (snapshot / "PRD_V1.1.md").write_bytes((PROJECT / "PRD_V1.1.md").read_bytes())
    configs_dir = snapshot / "configs"
    configs_dir.mkdir()
    run_records = []
    config_sources = sorted({filename for _, filename, _ in METHODS})
    for filename in config_sources:
        shutil.copy2(PROJECT / "configs" / filename, configs_dir / filename)
    for index, (name, filename, seed) in enumerate(METHODS, start=1):
        cfg = yaml.safe_load((configs_dir / filename).read_text())
        cfg["seed"] = seed
        cfg_path = configs_dir / f"run_{index:02d}_{name}_seed{seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        run_records.append(dict(index=index, name=name, seed=seed, config=str(cfg_path),
                                output=str(output / "runs" / f"{index:02d}_{name}_seed{seed}")))
    ae_config = configs_dir / "v1_r3_grct.yaml"
    plan = dict(version="1.1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        root=str(output), ae_config=str(ae_config), ae_output=str(output / "autoencoder"),
        source_hashes=source_hashes,
        config_hashes={str(path): _sha(path) for path in configs_dir.glob("*.yaml")},
        runs=run_records,
        gpu_policy=dict(min_free_mib=12000, max_utilization_percent=10, stable_observations=3,
                       observation_seconds=45),
        evaluation=dict(reconstruction_splits=["val_seen", "test_seen", "test_unseen_combo",
                                               "test_higher_complexity"],
                        paired_splits=["seen", "heldout"], ae_psnr_min=24.0))
    save_json(output / "plan.json", plan)
    save_json(output / "queue_status.json", dict(state="planned", plan=str(output / "plan.json"),
                                                  runs=len(run_records)))
    return output / "plan.json"


def available_gpu():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
                                 "--format=csv,noheader,nounits"], check=True,
                                capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    choices = []
    for row in result.stdout.splitlines():
        try:
            index, utilization, free = [int(value.strip()) for value in row.split(",")]
        except ValueError:
            continue
        if utilization <= 10 and free >= 12000:
            choices.append((free, index))
    return max(choices)[1] if choices else None


def _await_gpu(status_path, wait_for_gpu, policy):
    stable = 0
    while stable < policy["stable_observations"]:
        index = available_gpu()
        stable = stable+1 if index is not None else 0
        save_json(status_path, dict(state="waiting_for_resources" if not wait_for_gpu or stable < 3
                                    else "gpu_reserved", gpu=index, stable_observations=stable,
                                    required_observations=policy["stable_observations"],
                                    checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        if stable < policy["stable_observations"]:
            if not wait_for_gpu:
                return None
            time.sleep(policy["observation_seconds"])
    return index


def _ae_psnr(cfg, checkpoint, device):
    ae, _ = load_ae(checkpoint, device)
    values = []
    with torch.no_grad():
        for batch in make_loader(cfg, "val_seen"):
            x = batch["image"].to(device)
            values.extend(image_metrics(x, ae(x))["psnr"].cpu().tolist())
    return sum(values)/len(values)


def _require_source(plan):
    current = _source_manifest()
    if current != plan["source_hashes"]:
        changed = sorted(set(current) | set(plan["source_hashes"]),
                         key=lambda key: (current.get(key) == plan["source_hashes"].get(key), key))
        raise RuntimeError(f"Source changed after plan lock: {changed[:8]}")
    changed_configs = [path for path, digest in plan["config_hashes"].items()
                       if not Path(path).exists() or _sha(path) != digest]
    if changed_configs:
        raise RuntimeError(f"Locked configuration changed after plan creation: {changed_configs[:8]}")


def execute_plan(plan_path, wait_for_gpu=True):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text()); root = Path(plan["root"])
    status_path = root / "queue_status.json"
    lock_path = root / "queue.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another V1.1 queue owns {lock_path}") from error
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("state") == "finished":
            return
        try:
            _require_source(plan)
            gpu = _await_gpu(status_path, wait_for_gpu, plan["gpu_policy"])
            if gpu is None:
                return
            device = "cuda:0"
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            base_cfg = load_config(plan["ae_config"])
            from urbangene.data.synthetic import generate_manifests
            data_root = Path(base_cfg["data"]["root"])
            if not (data_root / "audit.json").exists():
                generate_manifests(base_cfg, data_root)
            ae_path = Path(plan["ae_output"]) / "best.pt"
            if not ae_path.exists():
                last = Path(plan["ae_output"]) / "last.pt"
                train_ae(base_cfg, plan["ae_output"], device, str(last) if last.exists() else None)
            score = _ae_psnr(base_cfg, ae_path, device)
            save_json(root / "ae_quality.json", dict(psnr=score, minimum=24.0,
                          passes=score >= 24.0, checkpoint=str(ae_path)))
            if score < 24.0:
                save_json(status_path, dict(state="blocked_ae_quality_gate", ae_psnr=score,
                    minimum_psnr=24.0, message="The frozen AE did not meet the predeclared readiness threshold"))
                return
            save_json(status_path, dict(state="training_models", gpu=gpu, ae_psnr=score,
                                        total_runs=len(plan["runs"])))
            checkpoints = []
            for record in plan["runs"]:
                cfg = load_config(record["config"]); run = Path(record["output"])
                checkpoint = run / "best.pt"
                if not checkpoint.exists():
                    last = run / "last.pt"
                    checkpoint = train(cfg, ae_path, run, device, str(last) if last.exists() else None)
                checkpoints.append(dict(name=record["name"], seed=record["seed"],
                                        checkpoint=str(checkpoint), run=str(run)))
                save_json(status_path, dict(state="training_models", gpu=gpu, ae_psnr=score,
                                            completed=len(checkpoints), total_runs=len(plan["runs"]),
                                            last_run=record["name"], last_seed=record["seed"]))
            save_json(root / "trained_checkpoints.json", checkpoints)
            for item in checkpoints:
                run = Path(item["run"])
                for split in plan["evaluation"]["reconstruction_splits"]:
                    metrics = run / "eval" / split / "metrics.json"
                    if not metrics.exists():
                        evaluate_reconstruction(item["checkpoint"], split, metrics.parent, device)
                paired = run / "eval/recombination/summary.json"
                if not paired.exists():
                    evaluate_recombination(item["checkpoint"], paired.parent, device, 128, item["seed"])
            _write_final_summary(root, checkpoints)
            save_json(status_path, dict(state="finished", gpu=gpu, ae_psnr=score,
                                        completed_runs=len(checkpoints)))
        except BaseException as error:
            save_json(status_path, dict(state="failed", error=repr(error),
                                        failed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            raise


def _write_final_summary(root, checkpoints):
    rows = []
    for item in checkpoints:
        run = Path(item["run"])
        metrics_path = run / "eval/test_seen/metrics.json"
        pair_path = run / "eval/recombination/summary.json"
        rows.append(dict(name=item["name"], seed=item["seed"],
                         reconstruction=json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
                         recombination=json.loads(pair_path.read_text()) if pair_path.exists() else None))
    save_json(root / "final_results.json", dict(rows=rows))
