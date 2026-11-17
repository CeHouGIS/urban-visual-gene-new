import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from urbangene_v2.backbone import FrozenVisualBackbone
from urbangene_v2.common import LEGACY_ROOT, OUTPUT_ROOT, writable
from urbangene_v2.data import FeatureDataset, ImageManifestDataset, _instance_mask
from urbangene_v2.extract import extract_manifest
from urbangene_v2.train import resolve_training_gpu_ids, train_discovery, prediction_from_slots


class FakeDinoForInterfaceTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 12, 4, stride=4)

    def forward_features(self, image):
        return {"x_norm_patchtokens": self.projection(image).flatten(2).transpose(1, 2)}


def test_frozen_backbone_contract():
    model = FrozenVisualBackbone(FakeDinoForInterfaceTest(), 16, 4, "torchhub_local",
                                 {"test_fixture": True})
    before = {k: value.clone() for k, value in model.state_dict().items()}
    model.train()
    assert not model.training and not model.model.training
    out = model(torch.rand(2, 3, 20, 25))
    assert out["features"].shape == (2, 12, 4, 4)
    assert out["coordinates"].shape == (4, 4, 2)
    assert not out["features"].requires_grad
    assert all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, before[k]) for k, value in model.state_dict().items())


def test_missing_backbone_is_not_replaced_by_random_weights():
    with pytest.raises(FileNotFoundError):
        FrozenVisualBackbone.from_config({"weights": "/does/not/exist", "family": "dinov3"})


def test_image_manifest_blocks_labels(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image_path)
    manifest = tmp_path / "images.jsonl"
    record = {"id": "test", "image": str(image_path), "split": "train"}
    manifest.write_text(json.dumps(record) + "\n")
    dataset = ImageManifestDataset(manifest)
    assert set(dataset[0]) == {"image", "id"}
    record["scene_spec"] = {"tree_count": 10}
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        ImageManifestDataset(manifest)


def test_mixed_image_sizes_extract_like_individual_backbone_calls(tmp_path, monkeypatch):
    import urbangene_v2.extract as extraction
    records = []
    rng = np.random.default_rng(7)
    for i, size in enumerate(((20, 25), (24, 24))):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(path)
        records.append({"id": str(i), "image": str(path), "split": "train"})
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    model = FrozenVisualBackbone(FakeDinoForInterfaceTest(), 16, 4, "torchhub_local",
                                 {"test_fixture": True})
    monkeypatch.setattr(extraction, "writable", lambda p: p)
    monkeypatch.setattr(FrozenVisualBackbone, "from_config", lambda cfg: model)
    calls = []
    cache = tmp_path / "features.npy"
    meta = extract_manifest(manifest, cache, {"resolution": 16}, batch_size=2,
                            progress_callback=lambda n, total: calls.append((n, total)))
    expected = torch.cat([model(ImageManifestDataset(manifest)[i]["image"][None])["features"]
                          for i in range(2)]).numpy()
    np.testing.assert_allclose(np.load(cache), expected, atol=1e-3, rtol=1e-3)
    assert calls == [(2, 2)] and meta["ids"] == ["0", "1"]
    assert meta["shape"] == [2, 12, 4, 4]


def test_feature_dataset_never_opens_annotations(tmp_path, monkeypatch):
    path = tmp_path / "features.npy"
    np.save(path, np.ones((3, 16, 12), dtype=np.float16))
    dataset = FeatureDataset(path, (4, 4))
    assert set(dataset[0]) == {"features"}
    assert dataset[0]["features"].shape == (12, 4, 4)
    with pytest.raises(ValueError):
        FeatureDataset(path, (3, 4))


@pytest.mark.parametrize("legacy", ["urbangene/outputs/v0/new", "urbangene_v11/outputs/new"])
def test_legacy_write_is_forbidden(legacy):
    with pytest.raises(ValueError):
        writable(LEGACY_ROOT / legacy)
    assert writable(OUTPUT_ROOT / "new").is_relative_to(OUTPUT_ROOT)


def test_eval_masks_have_visible_separate_trees():
    meta = {"building_shape": 0, "building_boxes": [[10, 20, 40, 90]],
            "tree_boxes": [[50, 70, 70, 112], [90, 70, 110, 112]]}
    mask = _instance_mask(meta, 256)
    assert set(np.unique(mask)) == {0, 1, 2, 3}
    assert mask[80, 60] == 2 and mask[80, 100] == 3


def test_training_gpu_selection_is_capped_at_two():
    assert resolve_training_gpu_ids(torch.device("cpu"), None, 0) == []
    assert resolve_training_gpu_ids(torch.device("cuda:0"), [0, 1], 4) == [0, 1]
    with pytest.raises(ValueError, match="capped at two"):
        resolve_training_gpu_ids(torch.device("cuda:0"), [0, 1, 2], 4)
    with pytest.raises(ValueError, match="primary CUDA device"):
        resolve_training_gpu_ids(torch.device("cuda:0"), [1, 0], 4)


def test_excess_gpu_request_fails_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    output = tmp_path / "must_not_be_created"
    config = {"seed": 42, "device": "cuda:0", "output": str(output),
              "training": {"gpu_ids": [0, 1, 2]}}
    with pytest.raises(ValueError, match="capped at two"):
        train_discovery(config)
    assert not output.exists()


def test_prediction_decodes_gates_and_background_without_renumbering_slots():
    masks = torch.tensor([[[.8, .1]], [[.1, .2]], [[.1, .7]]])
    exist = torch.tensor([9., -9., 9.])
    assert prediction_from_slots(masks, exist).tolist() == [[0, 3]]
    assert prediction_from_slots(masks, torch.full((3,), -9.)).tolist() == [[0, 0]]
    no_background = prediction_from_slots(masks, exist, background_slot=False)
    assert no_background.tolist() == [[1, 3]]
