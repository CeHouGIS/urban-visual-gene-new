"""Export real inference iterations of the existing DINOv3 slot pilot."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "object_centric_pilot"
VIS = Path("/root/.codex/visualizations/2026/10/09/01a11ea2-93da-7ff3-927a-a614c6a26c57")
sys.path.insert(0, str(PILOT))
from model import SlotModel  # noqa: E402


def data_image(image):
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def b64(array):
    return base64.b64encode(array.tobytes()).decode()


@torch.inference_mode()
def trace(model, features):
    group = model.group
    inputs = model.input_mlp(model.input_norm(features)) + model.position_strength * model.position(model.grid)[None]
    slots = group.mu + group.log_sigma.exp() * group.eval_noise
    keys, values = group.keys(group.norm_inputs(inputs)), group.values(group.norm_inputs(inputs))
    states, attentions, updates = [slots.clone()], [], []
    for _ in range(group.iterations):
        old = slots
        logits = torch.einsum("bkd,bpd->bkp", group.queries(group.norm_slots(slots)), keys) / (group.dim ** .5)
        attention = logits.softmax(dim=1)
        weights = attention + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        update = torch.einsum("bkp,bpd->bkd", weights, values)
        slots = group.gru(update.reshape(-1, group.dim), old.reshape(-1, group.dim)).reshape(1, group.slots, group.dim)
        slots = slots + group.mlp(group.norm_mlp(slots))
        attentions.append(attention[0].clone())
        updates.append(update[0].clone())
        states.append(slots.clone())
    reconstructed, masks = model.decode(slots)
    original = model(features)
    # A displayed trace must match the existing implementation, not a mockup.
    torch.testing.assert_close(slots, original["slots"], rtol=0, atol=0)
    torch.testing.assert_close(masks, original["masks"], rtol=0, atol=0)
    torch.testing.assert_close(attentions[-1][None], original["attention"], rtol=0, atol=0)
    all_weights = torch.stack(attentions + [masks[0]])
    torch.testing.assert_close(all_weights.sum(1), torch.ones((4, 784)), atol=1e-6, rtol=1e-6)
    return {"states": torch.cat(states).numpy(), "weights": all_weights.numpy(),
            "pooled_updates": torch.stack(updates).numpy(),
            "reconstructed_features": reconstructed[0].numpy()}


def main():
    torch.set_num_threads(4)
    HERE.mkdir(exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)
    archive = np.load(PILOT / "data/dinov3/dataset.npz")
    prepared_features = archive["features"]
    manifest = json.loads((PILOT / "data/dinov3/manifest.json").read_text())
    checkpoint_path = PILOT / "results/k32-refined/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SlotModel(slots=checkpoint["slots"], rgb_mode=checkpoint["rgb_mode"],
                      position_strength=checkpoint.get("position_strength", 1.)).eval()
    model.load_state_dict(checkpoint["model"])
    training = [i for i, r in enumerate(manifest["records"]) if r["split"] == "train"]
    pc = prepared_features[training, :, -3:][..., ::-1].astype(np.float32)
    low, high = np.quantile(pc, [.01, .99], axis=(0, 1))
    entries, records, raw = [], [], {}
    for city in ("London", "Austin", "Adelaide"):
        index = next(i for i, r in enumerate(manifest["records"]) if r["city"] == city and r["split"] != "train")
        record = manifest["records"][index]
        features = torch.from_numpy(prepared_features[index:index + 1].astype(np.float32))
        current = trace(model, features)
        for key, value in current.items():
            raw[f"{city}_{key}"] = value
        image = Image.open(PILOT / "data/dinov3" / record["image"]).convert("RGB")
        pcs = (features.numpy()[0, :, -3:][:, ::-1] - low) / (high - low)
        pc_image = Image.fromarray((pcs.clip(0, 1).reshape(28, 28, 3) * 255).astype(np.uint8))
        maps = current["weights"]
        hard = maps.argmax(1).astype(np.uint8)
        default = int(np.bincount(hard[-1], minlength=32).argmax())
        # Preserve real hard assignments separately; soft maps are quantized only
        # for compact display. Full precision is retained in trace.npz.
        entries.append({"city": city, "id": record["id"], "image": data_image(image),
                        "featureImage": data_image(pc_image.resize((448, 448), Image.Resampling.NEAREST)),
                        "weights": b64((maps * 255).round().astype(np.uint8)),
                        "hard": b64(hard), "states": current["states"].astype(np.float64).round(3).tolist(),
                        "maxWeights": maps.max(2).astype(np.float64).round(4).tolist(), "defaultUnit": default})
        records.append({"city": city, "id": record["id"], "split": record["split"],
                        "trace_equals_checkpoint": True, "default_unit_zero_based": default})
    np.savez_compressed(HERE / "trace.npz", **raw)
    payload = {"grid": 28, "slots": 32, "dimensions": 96, "scenes": entries}
    (VIS / "unit-construction-data.json").write_text(json.dumps(payload, separators=(",", ":")))
    provenance = {"checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "source": "existing DINOv3 PCA64 pilot; trained model inference on held-out examples",
                  "warning": "Candidate regions, not validated physical instances or a new model.",
                  "display": "soft maps uint8; hard maps from full precision; vectors rounded to .001; full precision trace.npz",
                  "iterations": 3, "scenes": records}
    (HERE / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps({"scenes": len(entries), "inline_data_bytes": (VIS / "unit-construction-data.json").stat().st_size,
                      "trace_matches_checkpoint": True}), flush=True)


if __name__ == "__main__":
    main()
