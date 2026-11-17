import json
import importlib.util
from pathlib import Path

import numpy as np
import torch

from urbangene_v2 import common
from urbangene_v2.scaling import train_epoch
from urbangene_v2.train import train_discovery


def test_gpu_picker_rejects_occupied_cards_and_caps_selection(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/run_real20k.py"
    spec = importlib.util.spec_from_file_location("real20k_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return "GPU-one\n"
        return "0, GPU-zero, 100, 0\n1, GPU-one, 100, 0\n2, GPU-two, 100, 0\n3, GPU-three, 100, 0\n"

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    assert [g["index"] for g in runner.idle_gpus()] == [0, 2]
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **kwargs:
                        "GPU-zero\nGPU-one\nGPU-two\nGPU-three\n"
                        if "--query-compute-apps=gpu_uuid" in command else query(command))
    assert runner.idle_gpus() == []


def test_resume_epoch_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    rng = np.random.default_rng(5)
    feature = tmp_path / "features.npy"
    np.save(feature, rng.normal(size=(4, 8, 3, 3)).astype(np.float16))
    config = {
        "seed": 42, "device": "cpu", "output": str(tmp_path / "uninterrupted"),
        "features": {"train": str(feature), "validation": str(feature)},
        "model": {"feature_dim": 8, "slot_dim": 8, "hidden_dim": 8, "max_slots": 3,
                  "iterations": 2, "spatial_strength": .1, "background_slot": True},
        "expected_counts": {"train": 4, "validation": 4},
        "training": {"epochs": 2, "batch_size": 2, "cpu_threads": 1,
                     "learning_rate": .0003, "gradient_clip": 1., "milestone_epochs": []}}
    train_discovery(config)
    expected = torch.load(Path(config["output"]) / "checkpoint_last.pt", weights_only=False)["model"]
    config = {**config, "output": str(tmp_path / "resumed")}
    first = train_epoch(config, [])
    assert first["epoch"] == 1 and first["steps"] == 2
    # The next worker may start with a completely different ambient RNG.
    torch.manual_seed(1234)
    np.random.seed(999)
    second = train_epoch(config, [])
    assert second["epoch"] == 2 and second["steps"] == 4
    actual = torch.load(Path(config["output"]) / "checkpoint_last.pt", weights_only=False)["model"]
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
    assert json.loads((Path(config["output"]) / "progress.json").read_text())["status"] == "complete"
