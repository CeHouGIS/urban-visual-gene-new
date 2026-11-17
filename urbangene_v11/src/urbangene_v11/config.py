import copy
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
V0 = PROJECT.parent / "urbangene"


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    m = cfg["model"]
    assert (m["gene_count"], m["max_instances"], m["latent_channels"],
            m["latent_hw"], m["gene_patch_hw"], m["position_grid"]) == (64, 16, 8, 32, 8, 16)
    assert m["budget_bits"] == 272 and m["scale_levels"] == [0.5, 1., 1.5]
    assert cfg["training"]["loss"] == "latent_l1"
    assert not m["allow_continuous_scene_residual"]
    c = m["composer"]
    assert c["instance_dim"] % c["relational_heads"] == 0
    assert c["instance_dim"] % c["spatial_heads"] == 0
    return cfg


def default_config():
    cfg = yaml.safe_load((V0 / "configs/v0_64genes.yaml").read_text())
    cfg["data"]["root"] = str((PROJECT / "outputs/v1_grct/data/full").resolve())
    cfg["data"]["workers"] = 0
    cfg["training"].update(amp=False, micro_batch_size=2, gradient_accumulation=16)
    cfg["model"]["composer"] = dict(type="grct", instance_dim=256,
             relational_layers=6, relational_heads=8, spatial_layers=4, spatial_heads=8,
             ffn_ratio=4, chunk_size=128, use_local_gene_features=True,
             residual_from_fixed_average=True, init_output_head_zero=True,
             cnn_width=64, cnn_depth=4)
    cfg["evaluation"] = dict(recombination_samples=128, overlap_threshold=0.5,
                             seeds=[42, 29, 43], ae_psnr_min=24., failure_rgb_l1=0.1,
                             validation_samples=2000, evaluation_batch_size=8)
    cfg["provenance"] = dict(version="1.1", baseline_root=str(V0), optional_paired_training=False)
    return cfg


def write_configs():
    base = default_config()
    variants = {"v1_r0_fixed": {"type": "fixed_average"},
                "v1_r1_alpha": {"type": "fixed_alpha"},
                "v1_r2_residual_cnn": {"type": "residual_cnn"},
                "v1_r3_grct": {}, "v1_r3_no_relation": {"relational_layers": 0},
                "v1_r3_no_spatial": {"use_local_gene_features": False}}
    for name, changes in variants.items():
        cfg = copy.deepcopy(base); cfg["model"]["composer"].update(changes)
        (PROJECT / f"configs/{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Match active trainable parameter counts against the full GRCT model.
    from urbangene.models.image_autoencoder import ImageAutoencoder
    from .model import UrbanGeneV11
    reference = UrbanGeneV11(base, ImageAutoencoder(8, base["ae"]["width"]))
    counts = reference.parameter_counts()
    r0p = copy.deepcopy(base)
    r0p["model"]["composer"]["type"] = "fixed_average"
    r0p["model"]["parser_parameter_target"] = counts["parser"]+counts["composer"]
    (PROJECT / "configs/v1_r0_parameter_matched.yaml").write_text(yaml.safe_dump(r0p, sort_keys=False))
    r2p = copy.deepcopy(base); r2p["model"]["composer"]["type"] = "residual_cnn"
    width = r2p["model"]["composer"]["cnn_width"]
    depth = r2p["model"]["composer"]["cnn_depth"]
    channels = r2p["model"]["latent_channels"]
    def cnn_parameters(w):
        total = (2*channels+1)*w*9+w
        total += (depth-1)*(w*w*9+w)
        return total+w*channels*9+channels
    width = min(range(8, 4097), key=lambda value: abs(cnn_parameters(value)-counts["composer"]))
    r2p["model"]["composer"]["cnn_width"] = width
    (PROJECT / "configs/v1_r2_parameter_matched.yaml").write_text(yaml.safe_dump(r2p, sort_keys=False))
    smoke = copy.deepcopy(base)
    smoke["data"].update(root=str((PROJECT / "outputs/v1_grct/smoke/data").resolve()), workers=0)
    smoke["data"]["counts"] = {k: 16 if k == "train" else 8 for k in smoke["data"]["counts"]}
    smoke["ae"].update(width=8, epochs=8, learning_rate=0.001)
    smoke["training"].update(epochs=12, batch_size=4, micro_batch_size=4, gradient_accumulation=1,
                              learning_rate=0.001, cpu_threads=4)
    smoke["model"].update(parser_width=16)
    smoke["model"]["composer"].update(instance_dim=32, relational_layers=2, relational_heads=4,
                   spatial_layers=2, spatial_heads=4, ffn_ratio=2, chunk_size=256, cnn_width=24, cnn_depth=2)
    smoke["evaluation"].update(recombination_samples=8, validation_samples=8)
    (PROJECT / "configs/smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
