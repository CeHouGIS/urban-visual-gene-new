import json
from pathlib import Path
import numpy as np
import pytest

from common import load_config, resume_compatible
from data import sample_ids, rank_ids, learning_rate
from prepare_data import select_grids, validate_records


def test_sample_resume_partition_and_epoch_boundary():
    full = sample_ids(97, 100, 32, 42)
    np.testing.assert_array_equal(full, np.concatenate([sample_ids(97, 100, 7, 42), sample_ids(104, 100, 25, 42)]))
    np.testing.assert_array_equal(np.concatenate([rank_ids(0, 100, 32, 42, r, 2) for r in range(2)]), sample_ids(0, 100, 32, 42))
    assert set(rank_ids(0, 100, 32, 42, 0, 2)).isdisjoint(rank_ids(0, 100, 32, 42, 1, 2))
    with pytest.raises(ValueError):
        rank_ids(0, 100, 31, 42, 0, 2)


def test_exact_final_partial_batch_and_lr():
    c = load_config(Path(__file__).resolve().parents[2] / 'configs/experiment.json')
    t = c['training']
    seen = updates = 0
    while seen < t['max_training_images']:
        batch = min(t['effective_batch_size'], t['max_training_images'] - seen)
        assert batch % 2 == 0
        seen += batch
        updates += 1
    assert seen == 1000000 and updates == 977 and batch == 576
    assert learning_rate(0, t) > 0
    assert learning_rate(t['warmup_updates'] * t['effective_batch_size'], t) == pytest.approx(t['learning_rate'])
    assert learning_rate(seen, t) == pytest.approx(t['minimum_learning_rate'])


def test_configuration_rejects_mismatched_budget(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'configs/experiment.json'
    c = json.loads(path.read_text())
    c['model']['multiscale_coefficient_budgets'][0] += 1
    target = tmp_path / 'bad.json'
    target.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='coefficient budgets'):
        load_config(target)


def test_microbatch_only_resume_preserves_global_training_semantics(tmp_path):
    from copy import deepcopy
    before = json.loads((Path(__file__).resolve().parents[2] / 'configs/experiment.json').read_text())
    after = deepcopy(before)
    after['training']['micro_batch_size'] = 192
    path = tmp_path / 'micro192.json'
    path.write_text(json.dumps(after))
    assert load_config(path) == after
    assert resume_compatible(before, after)
    after['training']['effective_batch_size'] *= 2
    assert not resume_compatible(before, after)
    after = deepcopy(before)
    after['model']['multiscale_coefficient_budgets'] = [256, 256, 512]
    assert not resume_compatible(before, after)


def test_geographic_selection_is_balanced_and_deterministic():
    rows = [{'continent': c, 'country': c+'country', 'ucdb_id': j, 'grid_id': f'{c}-{j}-{i}'}
            for c in ['A', 'B'] for j in [1, 2] for i in range(6)]
    selected = select_grids(rows, 8, 42)
    assert selected == select_grids(list(reversed(rows)), 8, 42)
    assert len(selected) == 8
    assert sum(x.startswith('A') for x in selected) == 4


def test_grid_validation_requires_five_complete_panorama_quartets():
    records = [{'grid_id': 'g', 'pano': f'p{i}', 'heading': h, 'id': f'p{i}_{h}'}
               for i in range(5) for h in [0, 90, 180, 270]]
    assert validate_records(records, 20) == ({'g'}, {f'p{i}' for i in range(5)})
    records[-1]['heading'] = 0
    with pytest.raises(ValueError, match='Incomplete grid'):
        validate_records(records, 20)
