import numpy as np
from scipy.optimize import linear_sum_assignment


def _match_iou(gt, pred):
    gt_ids = [int(x) for x in np.unique(gt) if x != 0]
    pred_ids = [int(x) for x in np.unique(pred) if x != 0]
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, gi in enumerate(gt_ids):
        gm = gt == gi
        for j, pj in enumerate(pred_ids):
            pm = pred == pj
            union = np.logical_or(gm, pm).sum()
            matrix[i, j] = np.logical_and(gm, pm).sum() / union if union else 0
    if matrix.size:
        rows, cols = linear_sum_assignment(1 - matrix)
        matches = [(gt_ids[r], pred_ids[c], float(matrix[r, c])) for r, c in zip(rows, cols)]
    else:
        matches = []
    return gt_ids, pred_ids, matrix, matches


def _comb2(value):
    return value * (value - 1) / 2


def _foreground_ari(gt, pred):
    gt, pred = gt[gt != 0], pred[gt != 0]
    if not len(gt):
        return 0.0
    _, gi = np.unique(gt, return_inverse=True)
    _, pi = np.unique(pred, return_inverse=True)
    table = np.zeros((gi.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(table, (gi, pi), 1)
    n = len(gt)
    sum_cells = _comb2(table).sum()
    sum_rows = _comb2(table.sum(1)).sum()
    sum_cols = _comb2(table.sum(0)).sum()
    total = _comb2(n)
    expected = sum_rows * sum_cols / total if total else 0
    maximum = .5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def instance_metrics(gt, pred):
    """Instance labels are consumed only here, outside training/model selection."""
    gt_ids, pred_ids, matrix, matches = _match_iou(gt, pred)
    matched_iou_sum = sum(row[2] for row in matches)
    split_gt = sum((matrix[i] > 0).sum() > 1 for i in range(len(gt_ids))) if pred_ids else 0
    merge_pred = sum((matrix[:, j] > 0).sum() > 1 for j in range(len(pred_ids))) if gt_ids else 0
    return {
        "gt_instances": len(gt_ids), "pred_instances": len(pred_ids),
        "matched_mask_iou": float(matched_iou_sum / max(len(gt_ids), len(pred_ids)))
        if gt_ids or pred_ids else 0.0,
        "foreground_ari": _foreground_ari(gt, pred),
        "count_abs_error": abs(len(gt_ids) - len(pred_ids)),
        "split_fraction": float(split_gt / len(gt_ids)) if gt_ids else 0.0,
        "merge_fraction": float(merge_pred / len(pred_ids)) if pred_ids else 0.0,
        "matches": matches,
    }


def stability_iou(first, flipped_back):
    first_ids, second_ids, _, matches = _match_iou(first, flipped_back)
    return float(sum(m[2] for m in matches) / max(len(first_ids), len(second_ids))) if matches else 0.0


def paired_bootstrap_difference(first, second, seed=187, samples=2000):
    """Image-level uncertainty for a paired validation comparison."""
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if difference.ndim != 1 or not len(difference) or not np.isfinite(difference).all():
        raise ValueError("Expected finite paired sample metrics")
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference), size=(samples, len(difference)))].mean(1)
    return {"mean_difference": float(difference.mean()),
            "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)),
            "resampling_unit": "image", "bootstrap_samples": samples, "seed": seed}
