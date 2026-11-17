import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    m = cfg["model"]
    assert cfg["data"]["image_size"] == 256, "V0 AE interface is 256 RGB -> 32 latent"
    assert m["latent_hw"] == 32 and not m["allow_continuous_scene_residual"]
    assert 1 <= m["gene_count"] <= m["max_gene_count"]
    assert 1 <= m["max_instances"] <= m["instance_limit"]
    assert m["position_grid"] >= 2 and all(s > 0 for s in m["scale_levels"])
    assert cfg["training"]["loss"] == "latent_l1", "V0 uses exactly one Gene loss"
    return cfg


def setup(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(path, cfg):
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def code_fingerprint():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})
