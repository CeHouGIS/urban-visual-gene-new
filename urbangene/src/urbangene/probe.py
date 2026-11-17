"""Supervised evaluation ONLY. Labels never train the Gene parser or AE."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .data.dataset import SceneDataset
from .data.synthetic import ALL_COMBOS, FACTORS
from .utils import save_checkpoint, save_json, setup, snapshot

NAMES = ["building_shape", "facade", "vegetation_layout", "road_width", "building_relation", "tree_count"]
CARDINALITIES = [3, 3, 3, 3, 3, 5]


def labels_for_spec(spec):
    return [spec[name]-(1 if name == "tree_count" else 0) for name in NAMES]


class AttributeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.SiLU(),
                                      nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
                                      nn.Conv2d(32, 48, 3, 2, 1), nn.SiLU(),
                                      nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
                                      nn.Linear(48*64, 128), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(128, n) for n in CARDINALITIES])

    def forward(self, x):
        y = self.features(x)
        return [h(y) for h in self.heads]


def train_probe(cfg, output, device="cpu"):
    setup(cfg["seed"], cfg["training"]["cpu_threads"])
    snapshot(output, cfg); model = AttributeProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # An independent calibration corpus, including all factor values/combinations
    # and repetition counts. Never fit a probe on the Gene test images. In
    # particular, counts 4/5 need training examples to be evaluated at all.
    import numpy as np
    def calibration_loader(namespace, count, shuffle):
        rng = np.random.default_rng(cfg["seed"]+namespace)
        specs = []
        for i in range(count):
            s = dict(zip(FACTORS, ALL_COMBOS[i % len(ALL_COMBOS)]))
            s.update(seed=cfg["seed"]*10**9+namespace*10**7+i, split="probe_calibration",
                     tree_count=1+(i//len(ALL_COMBOS)) % 5,
                     road_width=int(rng.integers(3)), building_relation=int(rng.integers(3)))
            specs.append(s)
        return DataLoader(SceneDataset(specs=specs), batch_size=cfg["training"]["batch_size"], shuffle=shuffle)
    train = calibration_loader(8, cfg.get("probe", {}).get("train_samples", 5000), True)
    val = calibration_loader(9, cfg.get("probe", {}).get("val_samples", 1000), False)
    save_json(Path(output) / "calibration_protocol.json", dict(
        purpose="supervised evaluation only; labels never used by Gene parser or AE",
        seed_namespaces=[8, 9], factors="all 27 triples; tree counts 1-5",
        train_samples=len(train.dataset), val_samples=len(val.dataset),
        reliable_per_factor_if_original_test_accuracy_at_least=0.85))
    best = -1.; history = []
    for epoch in range(cfg.get("probe", {}).get("epochs", 10)):
        model.train()
        for batch in train:
            labels = torch.tensor([labels_for_spec(train.dataset.specs[i]) for i in batch["index"].tolist()], device=device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch["image"].to(device))
            loss = sum(F.cross_entropy(pred, labels[:, j]) for j, pred in enumerate(preds))
            loss.backward(); optimizer.step()
        model.eval(); correct = torch.zeros(len(NAMES)); count = 0
        with torch.no_grad():
            for batch in val:
                labels = torch.tensor([labels_for_spec(val.dataset.specs[i]) for i in batch["index"].tolist()])
                preds = model(batch["image"].to(device))
                correct += torch.tensor([(p.cpu().argmax(1) == labels[:, j]).sum() for j, p in enumerate(preds)])
                count += len(labels)
        acc = (correct/count).tolist(); record = dict(epoch=epoch+1, val_accuracy=dict(zip(NAMES, acc)))
        history.append(record); print(record, flush=True)
        if sum(acc) > best:
            best = sum(acc)
            save_checkpoint(Path(output) / "best.pt", dict(model=model.state_dict(), config=cfg))
        save_json(Path(output) / "history.json", history)
        save_json(Path(output) / "status.json", dict(state="training", epoch=epoch+1))
    save_json(Path(output) / "status.json", dict(state="finished", epochs=len(history)))
    return Path(output) / "best.pt"


def load_probe(path, device):
    model = AttributeProbe().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval().requires_grad_(False), NAMES
