import json
from types import SimpleNamespace

from urbangene import schedule
from urbangene.budget import layout
from urbangene.utils import load_config


def test_busy_gpu_is_not_selected(monkeypatch):
    monkeypatch.setattr(schedule.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="0, 30000, 100\n1, 2000, 0\n2, 20000, 5\n"))
    assert schedule.available_gpu() == 2


def test_queue_waits_without_launching_training(cfg, tmp_path, monkeypatch):
    schedule.build_plan(cfg, tmp_path)
    monkeypatch.setattr(schedule, "available_gpu", lambda: None)
    monkeypatch.setattr(schedule, "train_ae", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must not train")))
    schedule.execute_plan(tmp_path / "plan.json", wait_for_gpu=False)
    assert json.loads((tmp_path / "queue_status.json").read_text())["state"] == "waiting_for_resources"


def test_full_matrix_budgets_and_heldout_configs(tmp_path):
    cfg = load_config("configs/v0_64genes.yaml")
    plan = schedule.build_plan(cfg, tmp_path)
    assert len(plan["runs"]) == 14
    for run in plan["runs"]:
        c = load_config(run["config"])
        assert layout(c, run["method"])["bits"] <= c["model"]["budget_bits"]
