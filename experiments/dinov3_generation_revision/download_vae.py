"""Fetch a public pretrained VAE and verify ModelScope's file checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


MODEL = "stabilityai/sd-vae-ft-mse"
REVISION = "79efec79e87756556b473a8991162904c8633521"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output):
    output.mkdir(parents=True, exist_ok=True)
    api = f"https://modelscope.cn/api/v1/models/{MODEL}/repo/files"
    with urllib.request.urlopen(api + "?" + urllib.parse.urlencode(
        {"Revision": REVISION, "Recursive": "true"}), timeout=30) as response:
        listing = json.load(response)
    assert listing["Success"], listing
    files = {entry["Path"]: entry for entry in listing["Data"]["Files"]}
    receipt = {"model": MODEL, "revision": REVISION,
               "provider": "ModelScope public mirror", "files": []}
    for name in ("config.json", "diffusion_pytorch_model.safetensors", "README.md"):
        entry, path = files[name], output / name
        if not path.exists() or sha256(path) != entry["Sha256"]:
            query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
            url = f"https://modelscope.cn/api/v1/models/{MODEL}/repo?{query}"
            temporary = path.with_suffix(path.suffix + ".partial")
            started = time.monotonic()
            print(f"Downloading {name}: {entry['Size']} bytes", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
            if temporary.stat().st_size != entry["Size"] or sha256(temporary) != entry["Sha256"]:
                raise ValueError(f"Downloaded file did not match authoritative checksum: {name}")
            temporary.replace(path)
            print(f"Verified {name} in {time.monotonic()-started:.1f}s", flush=True)
        receipt["files"].append({"path": name, "size": entry["Size"], "sha256": entry["Sha256"]})
    (output / "source.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models/sd-vae-ft-mse")
    download(parser.parse_args().output)
