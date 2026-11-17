import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT.parent / "outputs" / "v2"
LEGACY_ROOT = Path("/root/code/urban-visual-gene")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def writable(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"V2 outputs must be inside {OUTPUT_ROOT}: {path}")
    return path


def fresh_dir(path):
    path = writable(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if config.get("train_labels_for_instance_discovery") is not False:
        raise ValueError("Explicit no-label training flag is required")
    writable(config["output"])
    return config


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))}
