"""Choose one detector-labeled perspective view per global panorama."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .build_dataset import read_jsonl, validate_annotations, write_jsonl
from .curate_pilot import eligible
from .finalize_pilot import corrected_annotations


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
OFFSETS = ("0", "90", "180", "270")


def choose(candidates, proposals, config):
    by_scene = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate["scene_id"]].append(candidate)
    proposed = {row["image_id"]: row for row in corrected_annotations(proposals)}
    selected, pending, exhausted = [], [], []
    for scene, options in by_scene.items():
        options.sort(key=lambda row: OFFSETS.index(row["image_id"].rpartition("_")[2]))
        chosen = None
        for option in options:
            proposal = proposed.get(option["image_id"])
            if proposal is not None:
                chosen = eligible(proposal, config)
                if chosen is not None:
                    break
        if chosen is not None:
            selected.append(chosen)
            continue
        next_option = next((option for option in options if option["image_id"] not in proposed), None)
        if next_option is None:
            exhausted.append(scene)
        else:
            pending.append(next_option)
    return selected, pending, exhausted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DATA_ROOT / "global200_sources/candidates.jsonl")
    parser.add_argument("--proposal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "global200_sources/annotations.jsonl")
    parser.add_argument("--pending", type=Path, default=DATA_ROOT / "global200_sources/pending_candidates.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v0.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = read_jsonl(args.candidates)
    proposals = [row for path in args.proposal if path.exists() for row in read_jsonl(path)]
    selected, pending, exhausted = choose(candidates, proposals, config)
    validate_annotations(selected)
    write_jsonl(args.output, selected)
    write_jsonl(args.pending, pending)
    summary = {
        "panoramas": len({row["scene_id"] for row in candidates}),
        "eligible": len(selected), "pending": len(pending), "exhausted": len(exhausted),
        "categories": dict(Counter(obj["object_category"] for row in selected for obj in row["objects"])),
        "countries": len({row["country"] for row in selected}),
        "annotation_status": "machine_proposal_unreviewed",
        "selection": "first eligible heading in seeded 0/90/180/270 order",
    }
    (args.output.parent / "annotation_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
