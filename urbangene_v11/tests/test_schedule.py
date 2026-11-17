import json

from urbangene_v11.schedule import build_plan, execute_plan


def test_formal_queue_is_locked_and_waits_without_gpu(tmp_path, monkeypatch):
    plan_path = build_plan(tmp_path / "formal")
    plan = json.loads(plan_path.read_text())
    assert len(plan["runs"]) == 14
    assert len(plan["source_hashes"]) > 10 and len(plan["config_hashes"]) >= 15
    monkeypatch.setattr("urbangene_v11.schedule.available_gpu", lambda: None)
    execute_plan(plan_path, wait_for_gpu=False)
    status = json.loads((tmp_path / "formal/queue_status.json").read_text())
    assert status["state"] == "waiting_for_resources"
    assert not (tmp_path / "formal/autoencoder/best.pt").exists()
