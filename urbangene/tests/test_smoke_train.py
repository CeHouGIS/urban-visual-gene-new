import json

from urbangene.data.synthetic import generate_manifests
from urbangene.evaluate import evaluate
from urbangene.train import train_ae, train_gene


def test_end_to_end_checkpoint_and_unseen_evaluation(cfg, tmp_path):
    generate_manifests(cfg, cfg["data"]["root"])
    ae = train_ae(cfg, tmp_path / "ae")
    ckpt = train_gene(cfg, ae, tmp_path / "gene")
    metrics = evaluate(ckpt, "test_unseen_combo")
    assert metrics["samples"] == 8
    assert 0 < metrics["latent_l1"] < 100
    assert metrics["budget"]["bits"] == 128
    assert metrics["measured_program_file_bytes"] == 8*16
    assert (ckpt.parent / "eval/test_unseen_combo/recon_grid.png").exists()
    assert json.loads((ckpt.parent / "gradient_check.json").read_text())["parser"] > 0
