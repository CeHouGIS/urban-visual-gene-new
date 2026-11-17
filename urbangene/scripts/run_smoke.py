"""One command: tiny data -> AE -> UrbanGene/B0/B1/B2 -> explicit test evaluations."""
from pathlib import Path

from urbangene.data.synthetic import generate_manifests
from urbangene.evaluate import evaluate
from urbangene.train import train_ae, train_gene
from urbangene.utils import load_config, save_json

cfg = load_config("configs/smoke.yaml")
generate_manifests(cfg, cfg["data"]["root"])
root = Path("outputs/smoke")
ae = train_ae(cfg, root / "ae")
evaluate(ae, "val_seen")
for method in ["urbangene", "b0", "b1", "b2"]:
    ckpt = train_gene(cfg, ae, root / method, method)
    for split in ["val_seen", "test_seen", "test_unseen_combo", "test_higher_complexity"]:
        evaluate(ckpt, split)
save_json(root / "SMOKE_COMPLETE.json", dict(state="finished", scope="engineering only; 32 train images; not scientific evidence"))
