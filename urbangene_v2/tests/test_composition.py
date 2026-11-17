import pytest
import torch

from urbangene_v2.composition import SlotProgram, decode_program, edit_diagnostics
from urbangene_v2.discovery import DiscoveryModel


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(51)
    model = DiscoveryModel(12, 8, 5, 2, 12).eval()
    feature = torch.randn(1, 12, 4, 5)
    out = model(feature)
    program = SlotProgram.from_encoding(out, 4, 5, 'A')
    return model, out, program


def test_identity_and_serialized_replay_match_original_decoder(tmp_path):
    model, out, program = fixture()
    replay = decode_program(model.decoder, program)
    torch.testing.assert_close(replay['reconstructed_features'], out['reconstructed_features'])
    torch.testing.assert_close(replay['soft_masks'], out['soft_masks'])
    path = tmp_path / 'program.pt'
    program.save(path)
    loaded = SlotProgram.load(path)
    torch.testing.assert_close(decode_program(model.decoder, loaded)['reconstructed_features'],
                               replay['reconstructed_features'], rtol=0, atol=0)
    assert loaded.origins == program.origins


def test_delete_is_hard_removal_and_preserves_normalized_competition():
    model, _, program = fixture()
    removed = program.delete([2, 4])
    output = decode_program(model.decoder, removed)
    assert not output['soft_masks'][:, [2, 4]].any()
    torch.testing.assert_close(output['soft_masks'].sum(1), torch.ones(1, 4, 5))
    keep = [0, 1, 3]
    expected, _ = model.decoder(program.content[keep][None], program.geometry[keep][None],
                                program.existence[keep][None], 4, 5)
    torch.testing.assert_close(output['reconstructed_features'], expected)
    assert program.active.all()  # Editing does not mutate the source program.


def test_geometry_changes_without_content_mutation_and_capacity_is_enforced():
    _, _, program = fixture()
    moved = program.transform([1, 3], (.3, -.2), (1.5, .7))
    torch.testing.assert_close(moved.content, program.content, rtol=0, atol=0)
    torch.testing.assert_close(moved.geometry[[1, 3], :2], program.geometry[[1, 3], :2] + torch.tensor([.3, -.2]))
    with pytest.raises(ValueError, match='capacity exceeded'):
        program.duplicate([1])
    copied = program.delete([4]).duplicate([1], (.4, 0.))
    assert len(copied.content) == 5 and copied.active.all()
    torch.testing.assert_close(copied.content[4], program.content[1], rtol=0, atol=0)
    assert copied.origins[4] == program.origins[1]


def test_cross_scene_replace_uses_donor_content_and_keeps_other_recipient_slots():
    model, _, a = fixture()
    output = model(torch.ones(1, 12, 4, 5))
    b = SlotProgram.from_encoding(output, 4, 5, 'B')
    combined = a.replace_from([3], b, [2], (.2, .1), (.8, .9))
    torch.testing.assert_close(combined.content[3], b.content[2], rtol=0, atol=0)
    torch.testing.assert_close(combined.content[[0, 1, 2, 4]], a.content[[0, 1, 2, 4]], rtol=0, atol=0)
    assert combined.origins[3] == 'B:slot_2'
    assert torch.isfinite(decode_program(model.decoder, combined)['reconstructed_features']).all()


def test_foreground_slot_order_does_not_change_composition():
    model, _, program = fixture()
    order = [0, 3, 1, 4, 2]
    permuted = SlotProgram(program.content[order], program.geometry[order],
                           program.existence[order], program.active[order],
                           tuple(program.origins[i] for i in order), 4, 5)
    torch.testing.assert_close(decode_program(model.decoder, program)['reconstructed_features'],
                               decode_program(model.decoder, permuted)['reconstructed_features'])


@pytest.mark.parametrize('operation', ['delete_background', 'replace_background', 'inactive', 'bad_scale'])
def test_invalid_edits_are_rejected(operation):
    _, _, program = fixture()
    with pytest.raises(ValueError):
        if operation == 'delete_background':
            program.delete([0])
        elif operation == 'replace_background':
            program.replace_from([0], program, [1])
        elif operation == 'inactive':
            program.delete([1]).transform([1])
        else:
            program.transform([1], scale=(0., 1.))


def test_identity_metrics_do_not_invent_edit_success():
    model, _, program = fixture()
    original = decode_program(model.decoder, program)
    metrics, delta, _ = edit_diagnostics(original, original, [1], [1])
    assert metrics['feature_change_mse'] == 0 and not delta.any()
    assert metrics['change_energy_inside_support_fraction'] is None
    assert metrics['independent_ground_truth'] is False
