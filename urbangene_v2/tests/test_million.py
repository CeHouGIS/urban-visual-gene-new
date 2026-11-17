import itertools
import json

import numpy as np
import pytest
import torch

from urbangene_v2.million import (ShardedFeatures, ShuffledShardOrder, RepeatingShardOrder,
                                  accumulated_backward, scheduled_lr, feature_batches)


def test_shard_boundaries_shuffle_and_resume_cover_identical_samples(tmp_path):
    shards = []
    for i, (start, end) in enumerate([(0, 3), (3, 8), (8, 10)]):
        path = tmp_path / f"{i}.npy"
        np.save(path, np.arange(start, end, dtype=np.float16).reshape(-1, 1, 1, 1))
        shards.append({"path": str(path), "records": end - start})
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": 10, "shards": shards}))
    data = ShardedFeatures(index, max_open=1)
    assert [data[i].item() for i in range(10)] == list(range(10))
    assert len(data.cache) == 1
    with pytest.raises(IndexError):
        data[10]
    order = list(ShuffledShardOrder(data, 42, 2))
    assert sorted(order) == list(range(10))
    assert list(ShuffledShardOrder(data, 42, 2, skip=4)) == order[4:]
    continuous = list(itertools.islice(iter(RepeatingShardOrder(data, 42)), 25))
    resumed = list(itertools.islice(iter(RepeatingShardOrder(data, 42, epoch=1, skip=3)), 12))
    assert continuous[13:] == resumed
    threaded = list(feature_batches(data, iter(order), batch_size=3, workers=4))
    assert [len(batch) for batch in threaded] == [3, 3, 3, 1]
    assert torch.cat(threaded).flatten().tolist() == order


@pytest.mark.parametrize("corruption,message", [
    ("backbone", "backbone or shape"), ("manifest", "alignment"), ("feature", "checksum")])
def test_feature_index_rejects_wrong_model_images_or_content(tmp_path, monkeypatch, corruption, message):
    import importlib.util
    from pathlib import Path
    import sqlite3
    from urbangene_v2 import common

    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("million_runner_test", tools / "run_real1m.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "READY.json").write_text(json.dumps({"split_counts": {"train": 1}}))
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"records": []}))
    with sqlite3.connect(tmp_path / "verification.sqlite") as db:
        db.execute("CREATE TABLE accepted (id TEXT, sha256 TEXT, split TEXT)")
    manifest = tmp_path / "rgb_shards/train/images_0000000_0000001.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"id": "a", "image": "unused", "split": "train"}) + "\n")
    feature = tmp_path / "feature_shards/train/features_0000000_0000001.npy"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"checked bytes")
    meta = {"ids": ["a"], "manifest_sha256": common.sha256(manifest),
            "backbone": {"files": {"model.safetensors": common.sha256(weight_dir / "model.safetensors")}},
            "resolution": 448, "shape": [1, 1024, 28, 28], "dtype": "float16",
            "feature_sha256": common.sha256(feature)}
    if corruption == "backbone":
        meta["backbone"]["files"]["model.safetensors"] = "wrong model"
    elif corruption == "manifest":
        meta["ids"] = ["other photo"]
    else:
        meta["feature_sha256"] = "corrupted cache"
    feature.with_suffix(".npy.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        runner.build_indices({"data_root": str(tmp_path), "comparison_provenance": str(old),
                              "backbone": {"weights": str(weight_dir), "resolution": 448}})


def test_accumulation_matches_full_batch_gradient_even_with_short_micro_batch():
    # A deterministic scalar model isolates weighting from stochastic slot initialization.
    weight = torch.nn.Parameter(torch.tensor(2.))
    features = torch.arange(7, dtype=torch.float32) / 7

    def model(x):
        return {"reconstructed_features": x * weight}

    full_loss = ((model(features)["reconstructed_features"] - features) ** 2).mean()
    full_loss.backward()
    expected = weight.grad.clone()
    weight.grad = None
    actual_loss = accumulated_backward(model, features, effective_batch=7, micro_batch=3)
    torch.testing.assert_close(weight.grad, expected)
    assert actual_loss == pytest.approx(float(full_loss.detach()), rel=1e-6)
    with pytest.raises(ValueError):
        accumulated_backward(model, features, effective_batch=8, micro_batch=3)


def test_warmup_cosine_schedule_has_expected_endpoints():
    cfg = {"learning_rate": .0003, "minimum_learning_rate": .00003,
           "warmup_updates": 2000, "max_updates": 100000}
    assert scheduled_lr(cfg, 0) == pytest.approx(.0003 / 2000)
    assert scheduled_lr(cfg, 1999) == pytest.approx(.0003)
    assert scheduled_lr(cfg, 2000) == pytest.approx(.0003)
    assert scheduled_lr(cfg, 100000) == pytest.approx(.00003)
