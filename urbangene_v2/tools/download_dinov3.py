#!/usr/bin/env python3
"""Download and verify the gated DINOv3 ViT-L/16 Hugging Face snapshot."""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModel


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REVISION = "main"
MODEL_DIR = ROOT.parent / "models" / "dinov3-vitl16"
HF_HOME = ROOT.parent / "models" / "hf_home"
METADATA = ROOT.parent / "outputs" / "v2" / "dinov3_download.json"


def read_token(env_path):
    if not env_path.is_file():
        raise FileNotFoundError(f"Create {env_path} and set HF_TOKEN")
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value and value != "paste_your_read_token_here":
                return value
            break
    raise ValueError(f"HF_TOKEN is empty in {env_path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    token = read_token(ROOT / ".env")
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    HF_HOME.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty model directory: {MODEL_DIR}")

    info = HfApi(token=token).model_info(REPO_ID, revision=REVISION)
    snapshot = Path(snapshot_download(
        repo_id=REPO_ID,
        revision=info.sha,
        token=token,
        local_dir=MODEL_DIR,
    ))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if "dinov3" not in config.model_type.lower():
        raise ValueError(f"Unexpected model_type: {config.model_type}")
    if config.patch_size != 16 or config.hidden_size != 1024:
        raise ValueError(
            f"Unexpected ViT-L/16 dimensions: patch={config.patch_size}, "
            f"hidden={config.hidden_size}"
        )

    model, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Downloaded checkpoint did not load completely into AutoModel")
    model.eval().requires_grad_(False)
    del model

    files = {}
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = str(path.relative_to(snapshot))
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise FileNotFoundError("Snapshot contains no model weight files")
    metadata = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "model_type": config.model_type,
        "architecture": config.architectures,
        "patch_size": config.patch_size,
        "hidden_size": config.hidden_size,
        "model_dir": str(snapshot),
        "loaded_with_transformers": True,
        "missing_keys": loading.get("missing_keys", []),
        "mismatched_keys": loading.get("mismatched_keys", []),
        "files": files,
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "status": "downloaded_and_loaded",
        "repo_id": REPO_ID,
        "revision": info.sha,
        "model_dir": str(snapshot),
        "model_type": config.model_type,
        "weight_files": [name for name in files if name.endswith((".safetensors", ".bin"))],
        "metadata": str(METADATA),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        try:
            secret = read_token(Path(__file__).resolve().parents[1] / ".env")
            if secret:
                message = message.replace(secret, "[REDACTED]")
        except Exception:
            pass
        raise SystemExit(f"DINOv3 download failed: {message}") from None
