"""Serial, resumable pipeline. Resource waits never allocate a GPU."""
import copy
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from .data.synthetic import generate_manifests
from .evaluate import evaluate
from .train import train_ae, train_gene
from .utils import code_fingerprint, digest, save_json


def build_plan(cfg, root):
    root = Path(root); config_dir = root / "configs"; config_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    # Start with central capacity; only run the rest after its validation gate passes.
    cases = [("urbangene", 64, 16)]
    cases += [(m, 64, 16) for m in ["b0", "b1", "b2"]]
    cases += [("urbangene", k, n) for k in [32, 64, 128] for n in [8, 16, 32] if (k, n) != (64, 16)]
    for method, k, n in cases:
        c = copy.deepcopy(cfg); c["model"].update(gene_count=k, max_instances=n)
        # ceil(log2 K) + enable(1) + position(8) + scale(2).
        from .budget import width
        c["model"]["budget_bits"] = n*(width(k)+11)
        name = f"{method}_k{k}_m{n}_seed{c['seed']}"
        path = config_dir / f"{name}.yaml"; path.write_text(yaml.safe_dump(c, sort_keys=False))
        entries.append(dict(name=name, method=method, config=str(path), cfg_hash=digest(c)))
    for grid in [8, 32]:
        c = copy.deepcopy(cfg); c["model"]["position_grid"] = grid
        c["model"]["budget_bits"] = 16*(width(64)+1+width(grid**2)+2)
        name = f"urbangene_k64_m16_grid{grid}_seed{cfg['seed']}"
        path = config_dir / f"{name}.yaml"; path.write_text(yaml.safe_dump(c, sort_keys=False))
        entries.append(dict(name=name, method="urbangene", config=str(path), cfg_hash=digest(c)))
    plan = dict(base_config=cfg, runs=entries, code_fingerprint=code_fingerprint(),
                execution_order="AE -> validation gate -> main -> validation gate -> baselines/capacity/precision -> lock -> probe -> tests",
                formal_gene_runs=len(entries),
                gates=dict(ae_psnr_min=24., main_relative_shuffle_improvement_min=0.01),
                thresholds="engineering readiness gates chosen before formal results; not scientific success criteria",
                precision="AE may use bf16; Gene uses float32 for discrete renderer gradients",
                test_policy="test splits only after all configs/checkpoints are locked using val_seen")
    save_json(root / "plan.json", plan)
    return plan


def available_gpu(min_free_mib=12000, max_utilization=10):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        idx, free, util = [int(s.strip()) for s in line.split(",")]
        if free >= min_free_mib and util <= max_utilization:
            return idx
    return None


def execute_plan(plan_path, wait_for_gpu=True, device=None, poll_seconds=45):
    from .utils import load_config
    plan_path = Path(plan_path).resolve(); root = plan_path.parent
    plan = json.loads(plan_path.read_text()); cfg = plan["base_config"]
    assert plan["code_fingerprint"] == code_fingerprint(), "Code changed since plan creation; regenerate and review plan"
    status_path = root / "queue_status.json"
    # A single local queue, no duplicate workers. Does not claim exclusivity across hosts.
    import fcntl
    with (root / "queue.lock").open("w") as lock:
        try:
            assert plan["code_fingerprint"] == code_fingerprint(), "Code changed while queue was waiting"
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This training queue already has an active worker")
        while device is None:
            if plan["code_fingerprint"] != code_fingerprint():
                save_json(status_path, dict(state="failed", reason="Code changed while waiting; regenerate plan"))
                return
            free_gb = shutil.disk_usage(root).free/1024**3
            gpu = available_gpu() if free_gb >= 2 else None
            if gpu is not None:
                # Require three idle observations to avoid starting on a transient dip.
                observations = [gpu]
                for _ in range(2):
                    save_json(status_path, dict(state="checking_gpu_idle", gpu=gpu, observations=len(observations)))
                    time.sleep(min(poll_seconds, 45))
                    observations.append(available_gpu())
                if all(g == gpu for g in observations):
                    device = f"cuda:{gpu}"
                    break
            save_json(status_path, dict(state="waiting_for_resources", free_disk_gib=round(free_gb, 2),
                      requirement="GPU <=10% utilization, >=12GiB free, three observations; disk >=2GiB",
                      pid=__import__("os").getpid(), last_checked_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                      poll_seconds=poll_seconds))
            if not wait_for_gpu:
                return
            time.sleep(poll_seconds)
        try:
            assert plan["code_fingerprint"] == code_fingerprint(), "Code changed while waiting"
            generate_manifests(cfg, cfg["data"]["root"])
            ae_dir = root / "ae"
            ae_path = ae_dir / "best.pt"
            def complete(folder):
                p = folder / "status.json"
                return p.exists() and json.loads(p.read_text()).get("state") == "finished"
            if not complete(ae_dir):
                save_json(status_path, dict(state="training_ae", device=device))
                train_ae(cfg, ae_dir, device, ae_dir / "last.pt" if (ae_dir / "last.pt").exists() else None)
            ae_metrics = evaluate(ae_path, "val_seen", device)
            if ae_metrics["psnr"] < plan["gates"]["ae_psnr_min"]:
                save_json(status_path, dict(state="diagnosis_required", stage="ae", metrics=ae_metrics,
                          reason="Frozen AE is below the predeclared reconstruction readiness threshold"))
                return
            for idx, entry in enumerate(plan["runs"]):
                assert plan["code_fingerprint"] == code_fingerprint(), "Code changed during training"
                c = load_config(entry["config"]); assert digest(c) == entry["cfg_hash"]
                folder = root / entry["name"]
                save_json(status_path, dict(state="training_gene", run=entry["name"], device=device))
                if not complete(folder):
                    train_gene(c, ae_path, folder, entry["method"], device,
                               folder / "last.pt" if (folder / "last.pt").exists() else None)
                metric = evaluate(folder / "best.pt", "val_seen", device)
                if idx == 0:
                    shuffled = metric["shuffled_latent_l1"]
                    improvement = 1-metric["latent_l1"]/max(shuffled, 1e-9)
                    if improvement < plan["gates"]["main_relative_shuffle_improvement_min"]:
                        save_json(status_path, dict(state="diagnosis_required", stage="main_gene", metrics=metric,
                                  reason="Correct program does not improve over shuffled programs by 1%; stop matrix"))
                        return
            save_json(root / "TEST_PROTOCOL_LOCK.json", dict(plan_sha256=__import__("hashlib").sha256(plan_path.read_bytes()).hexdigest(),
                      selection="lowest full val_seen latent L1 per fixed run", runs=plan["runs"]))
            from .probe import train_probe
            probe_path = root / "probe" / "best.pt"
            if not complete(root / "probe"):
                save_json(status_path, dict(state="training_evaluation_probe", device=device))
                train_probe(cfg, root / "probe", device)
            rows = []
            evaluate(ae_path, "test_seen", device)
            for entry in plan["runs"]:
                for split in ["test_seen", "test_unseen_combo", "test_higher_complexity"]:
                    metric = evaluate(root / entry["name"] / "best.pt", split, device, probe_path=probe_path)
                    rows.append({k: v for k, v in metric.items() if not isinstance(v, (dict, list))})
            with (root / "results.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
                writer.writeheader(); writer.writerows(rows)
            save_json(status_path, dict(state="finished", formal_runs=len(plan["runs"]), device=device,
                      limitations="one training seed; reconstruction only; probe must be calibrated per test factor"))
        except Exception as e:
            save_json(status_path, dict(state="failed", error=repr(e), device=device))
            raise
