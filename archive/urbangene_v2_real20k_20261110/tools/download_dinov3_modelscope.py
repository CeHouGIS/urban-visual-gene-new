#!/usr/bin/env python3
"""Fetch a pinned ModelScope snapshot without reading HF credentials."""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from download_dinov3 import MODEL_DIR, METADATA, sha256


REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REVISION = "99b6205b0321c4d9726dd072d8c6a80aa6be72df"
API = f"https://modelscope.cn/api/v1/models/{REPO_ID}"
EXPECTED_WEIGHT_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"


def request(path, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(API + path, headers=headers), timeout=45)


def download_file(entry):
    name = entry["Path"]
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Unsafe remote path: {name}")
    target = MODEL_DIR / name
    expected_size, expected_hash = entry["Size"], entry["Sha256"]
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != expected_hash:
            raise FileExistsError(f"Existing file differs; refusing overwrite: {target}")
        print(f"Already verified: {name}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    path = "/repo?" + urllib.parse.urlencode({
        "Revision": REVISION, "FilePath": name})
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Partial download is too large: {partial}")
        if offset == expected_size:
            break
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with request(path, **headers) as response:
                status = response.status
                if offset and status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an incorrect resume range")
                elif offset and status == 200:
                    offset = 0
                elif status != 200:
                    raise ValueError(f"Unexpected download HTTP status {status}")
                count, report_time = offset, time.monotonic()
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        count += len(chunk)
                        if count > expected_size:
                            raise ValueError("Download exceeds repository file size")
                        if time.monotonic() - report_time >= 15:
                            print(f"{name}: {count / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
                            report_time = time.monotonic()
            if partial.stat().st_size != expected_size:
                raise OSError("Download ended before expected file size")
            break
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            print(f"Retry {attempt + 1} for {name}: {error}", flush=True)
    if sha256(partial) != expected_hash:
        raise ValueError(f"SHA256 mismatch: {partial}; file not accepted")
    partial.replace(target)
    print(f"Verified: {name} ({expected_size} bytes)", flush=True)


def main():
    # This route is public and never opens .env or passes HF credentials.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    torch.set_num_threads(4)
    with request("/repo/files?" + urllib.parse.urlencode({
            "Revision": REVISION, "Recursive": "true"})) as response:
        inventory = json.load(response)
    if inventory.get("Code") != 200:
        raise ValueError(f"Could not list pinned snapshot: {inventory.get('Message')}")
    entries = [row for row in inventory["Data"]["Files"] if row["Type"] == "blob"]
    by_name = {row["Path"]: row for row in entries}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md"):
        if name not in by_name:
            raise FileNotFoundError(f"Snapshot lacks {name}")
    if by_name["model.safetensors"]["Sha256"] != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Pinned weight identity differs from the inspected repository")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    inventory_path = METADATA.with_name("dinov3_modelscope_inventory.json")
    inventory_path.write_text(json.dumps({
        "source": "modelscope", "repo_id": REPO_ID, "revision": REVISION,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries}, indent=2) + "\n")
    # Small files first so model configuration is available during download.
    for entry in sorted(entries, key=lambda row: row["Size"]):
        download_file(entry)

    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    if (config.model_type != "dinov3_vit" or config.patch_size != 16
            or config.hidden_size != 1024 or config.num_hidden_layers != 24):
        raise ValueError("Snapshot is not the expected DINOv3 ViT-L/16")
    model, loading = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, output_loading_info=True,
        low_cpu_mem_usage=True)
    if any(loading.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete checkpoint loading: {loading}")
    model.eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    files = {row["Path"]: {"size_bytes": row["Size"], "sha256": row["Sha256"]}
             for row in entries}
    metadata = {
        "source": "modelscope", "repo_id": REPO_ID, "revision": REVISION,
        "source_url": f"https://modelscope.cn/models/{REPO_ID}",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": config.model_type, "architecture": config.architectures,
        "patch_size": config.patch_size, "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_dir": str(MODEL_DIR), "loaded_with_transformers": True,
        "loading_info": loading, "image_processor": type(processor).__name__,
        "device": "cpu", "files": files,
        "integrity_note": "Hashes checked against the pinned ModelScope snapshot; not independently compared with Hugging Face.",
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "downloaded_and_loaded", "source": "modelscope",
                      "repo_id": REPO_ID, "revision": REVISION,
                      "model_dir": str(MODEL_DIR), "metadata": str(METADATA)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
