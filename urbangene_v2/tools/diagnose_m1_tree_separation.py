#!/usr/bin/env python3
"""Evaluate repeated-tree separation; annotations are used only in diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    directory = Path(args.evaluation)
    metrics = json.loads((directory / "metrics.json").read_text())
    rows = []
    for index in range(metrics["samples"]):
        data = np.load(directory / f"sample_{index:04d}.npz")
        gt, prediction = data["gt"], data["prediction"]
        # The renderer assigns buildings 1,2; tree identities start at 3.
        trees = [int(i) for i in np.unique(gt) if i >= 3]
        objects = [int(i) for i in np.unique(prediction) if i != 0]
        matched = {int(gi): (int(pi), float(iou)) for gi, pi, iou in metrics["per_sample"][index]["matches"]}
        split, merge = 0, 0
        for tree in trees:
            support = gt == tree
            split += sum((np.logical_and(support, prediction == pi).sum() / support.sum()) >= .1 for pi in objects) > 1
        for pi in objects:
            support = prediction == pi
            covered = sum((np.logical_and(support, gt == tree).sum() / (gt == tree).sum()) >= .1 for tree in trees)
            merge += covered > 1
        feature = data["features"].astype(np.float64).transpose(1, 2, 0)
        normalized = feature / np.linalg.norm(feature, axis=-1, keepdims=True).clip(1e-12)
        prototypes = [normalized[gt == tree].mean(0) for tree in trees]
        if len(prototypes) > 1:
            p = np.stack(prototypes)
            p /= np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-12)
            cosine = p @ p.T
            pair_cosine = float(cosine[np.triu_indices(len(p), 1)].mean())
        else:
            pair_cosine = None
        rows.append({
            "index": index, "visible_trees_on_patch_grid": len(trees),
            "tree_pixels": {str(t): int((gt == t).sum()) for t in trees},
            "matched_tree_iou_mean": float(np.mean([matched.get(tree, (0, 0.))[1] for tree in trees])) if trees else 0.,
            "distinct_tree_recall_iou50": float(np.mean([matched.get(tree, (0, 0.))[1] >= .5 for tree in trees])) if trees else 0.,
            "tree_split_fraction_10pct": float(split / len(trees)) if trees else 0.,
            "predicted_groups_merging_multiple_trees_10pct": int(merge),
            "mean_cosine_between_gt_tree_feature_prototypes": pair_cosine,
        })
    fields = ("visible_trees_on_patch_grid", "matched_tree_iou_mean", "distinct_tree_recall_iou50", "tree_split_fraction_10pct", "predicted_groups_merging_multiple_trees_10pct", "mean_cosine_between_gt_tree_feature_prototypes")
    result = {"scope": "Evaluation only. Renderer tree identities never enter discovery training.",
              "metrics_source": str(directory / "metrics.json"),
              "definition": "Tree IoU/recall use the same one-to-one Hungarian assignment as full-scene evaluation, including buildings; split/merge use at least 10% of each GT tree area.",
              "feature_contrast_note": "GT-pooled tree prototypes diagnose feature similarity; these are evaluation diagnostics, not unsupervised predictions.",
              "means": {field: float(np.mean([row[field] for row in rows if row[field] is not None])) for field in fields},
              "per_sample": rows}
    with (directory / "tree_separation.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["means"], indent=2))


if __name__ == "__main__":
    main()
