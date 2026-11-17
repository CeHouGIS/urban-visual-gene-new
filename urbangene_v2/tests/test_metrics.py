import numpy as np

from urbangene_v2.metrics import instance_metrics, stability_iou, paired_bootstrap_difference


def test_perfect_segmentation_and_permutation():
    gt = np.array([[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8)
    pred = gt.copy()
    pred[gt == 1], pred[gt == 2] = 7, 8
    metrics = instance_metrics(gt, pred)
    assert metrics["matched_mask_iou"] == 1
    assert metrics["foreground_ari"] == 1
    assert metrics["count_abs_error"] == 0
    assert metrics["split_fraction"] == metrics["merge_fraction"] == 0
    assert stability_iou(gt, pred) == 1


def test_collapsed_slot_and_empty_prediction_are_failures():
    gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8)
    merged = np.ones_like(gt)
    metrics = instance_metrics(gt, merged)
    assert metrics["foreground_ari"] == 0
    assert metrics["merge_fraction"] == 1
    assert metrics["count_abs_error"] == 1
    empty = instance_metrics(gt, np.zeros_like(gt))
    assert empty["matched_mask_iou"] == 0
    assert empty["count_abs_error"] == 2


def test_stability_penalizes_unmatched_instances():
    first = np.array([[1, 1, 0, 0], [0, 0, 0, 0]])
    second = np.array([[4, 4, 0, 0], [0, 0, 7, 7]])
    assert stability_iou(first, second) == .5


def test_paired_bootstrap_constant_improvement():
    result = paired_bootstrap_difference([.4, .7, .6], [.2, .5, .4])
    assert np.isclose(result["mean_difference"], .2)
    assert np.isclose(result["ci95_low"], .2)
    assert np.isclose(result["ci95_high"], .2)
