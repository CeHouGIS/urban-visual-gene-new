"""Edits of continuous discovery slots; not the discrete Gene scene program."""
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SlotProgram:
    content: torch.Tensor
    geometry: torch.Tensor
    existence: torch.Tensor
    active: torch.Tensor
    origins: tuple[str, ...]
    height: int
    width: int
    background_slot: bool = True

    def __post_init__(self):
        m, _ = self.content.shape
        if (self.geometry.shape != (m, 4) or self.existence.shape != (m,)
                or self.active.shape != (m,) or self.active.dtype != torch.bool
                or len(self.origins) != m or min(self.height, self.width, m) < 1):
            raise ValueError('Invalid slot program dimensions')
        tensors = [self.content, self.geometry, self.existence, self.active]
        if len({t.device for t in tensors}) != 1:
            raise ValueError('Program tensors must share a device')
        if not all(torch.isfinite(t).all() for t in tensors[:3]):
            raise ValueError('Program tensors must be finite')
        if (self.geometry[:, 2:] <= 0).any() or not self.active.any():
            raise ValueError('Positive scales and at least one active slot are required')
        if self.background_slot and not self.active[0]:
            raise ValueError('The background slot must remain active')

    @classmethod
    def from_encoding(cls, output, height, width, source, background_slot=True):
        if output['slot_features'].shape[0] != 1:
            raise ValueError('Encode one source image per editable program')
        content = output['slot_features'][0].detach().clone()
        return cls(content, output['estimated_geometry'][0].detach().clone(),
                   output['exist_logits'][0].detach().clone(),
                   torch.ones(len(content), dtype=torch.bool, device=content.device),
                   tuple(f'{source}:slot_{i}' for i in range(len(content))),
                   height, width, background_slot)

    def clone(self):
        return replace(self, content=self.content.clone(), geometry=self.geometry.clone(),
                       existence=self.existence.clone(), active=self.active.clone())

    def to(self, device):
        return replace(self, **{name: getattr(self, name).to(device)
                               for name in ['content', 'geometry', 'existence', 'active']})

    def _indices(self, indices, allow_background=False):
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Specify distinct slot indices')
        for i in indices:
            if not isinstance(i, int) or not 0 <= i < len(self.content):
                raise ValueError('Slot index is outside the fixed program capacity')
            if self.background_slot and i == 0 and not allow_background:
                raise ValueError('Editing the background slot is disabled')
            if not self.active[i]:
                raise ValueError('Cannot edit an inactive source slot')
        return indices

    def delete(self, indices):
        indices = self._indices(indices)
        changed = self.clone()
        changed.active[indices] = False
        return replace(changed)  # Revalidate, including the all-inactive case.

    def transform(self, indices, translation=(0., 0.), scale=(1., 1.)):
        indices = self._indices(indices)
        translation = self.geometry.new_tensor(translation)
        scale = self.geometry.new_tensor(scale)
        if (translation.shape != (2,) or scale.shape != (2,)
                or not torch.isfinite(translation).all() or not torch.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError('Use finite XY translation and two positive scale factors')
        changed = self.clone()
        changed.geometry[indices, :2] += translation
        changed.geometry[indices, 2:] *= scale
        return replace(changed)

    def replace_from(self, target_indices, donor, donor_indices, translation=(0., 0.), scale=(1., 1.)):
        targets = list(target_indices)
        sources = donor._indices(donor_indices)
        if (len(targets) != len(sources) or len(set(targets)) != len(targets)
                or self.content.shape[1] != donor.content.shape[1]
                or self.content.device != donor.content.device):
            raise ValueError('Replacement needs compatible programs and one-to-one slot indices')
        for i in targets:
            if not isinstance(i, int) or not 0 <= i < len(self.content):
                raise ValueError('Replacement exceeds fixed program capacity')
            if self.background_slot and i == 0:
                raise ValueError('Replacing the background slot is disabled')
        moved = donor.transform(sources, translation, scale)
        changed, origins = self.clone(), list(self.origins)
        changed.content[targets] = moved.content[sources]
        changed.geometry[targets] = moved.geometry[sources]
        changed.existence[targets] = moved.existence[sources]
        changed.active[targets] = True
        for target, source in zip(targets, sources):
            origins[target] = donor.origins[source]
        return replace(changed, origins=tuple(origins))

    def duplicate(self, indices, translation=(0., 0.), scale=(1., 1.)):
        sources = self._indices(indices)
        free = (~self.active).nonzero().flatten().tolist()
        if len(free) < len(sources):
            raise ValueError('Program capacity exceeded; explicitly free slots before duplication')
        return self.replace_from(free[:len(sources)], self, sources, translation, scale)

    def save(self, path):
        value = {'format': 'urbangene.continuous_slots.v1',
                 'content': self.content.detach().cpu(), 'geometry': self.geometry.detach().cpu(),
                 'existence': self.existence.detach().cpu(), 'active': self.active.detach().cpu(),
                 'origins': list(self.origins), 'height': self.height, 'width': self.width,
                 'background_slot': self.background_slot,
                 'discrete_gene_program': False}
        torch.save(value, Path(path))

    @classmethod
    def load(cls, path, device='cpu'):
        value = torch.load(path, map_location=device, weights_only=True)
        if value.pop('format') != 'urbangene.continuous_slots.v1':
            raise ValueError('Unknown slot program format')
        value.pop('discrete_gene_program')
        value['origins'] = tuple(value['origins'])
        return cls(**value)


def decode_program(decoder, program):
    """Recompute masks from slots and geometry; no source feature/mask input."""
    if decoder.background_slot != program.background_slot:
        raise ValueError('Decoder and program disagree about background handling')
    indices = program.active.nonzero().flatten()
    features, selected_masks = decoder(program.content[indices][None],
                                       program.geometry[indices][None],
                                       program.existence[indices][None],
                                       program.height, program.width)
    masks = selected_masks.new_zeros((1, len(program.content), program.height, program.width))
    masks[:, indices] = selected_masks
    return {'reconstructed_features': features, 'soft_masks': masks}


@torch.no_grad()
def edit_diagnostics(baseline, edited, source_indices, edited_indices):
    """Feature-space locality diagnostics, not semantic or RGB edit accuracy."""
    before, after = baseline['reconstructed_features'], edited['reconstructed_features']
    delta = (after - before).square().mean(1)[0]
    before_ids = baseline['soft_masks'].argmax(1)[0]
    after_ids = edited['soft_masks'].argmax(1)[0]
    support = torch.zeros_like(before_ids, dtype=torch.bool)
    for i in source_indices:
        support |= before_ids == i
    for i in edited_indices:
        support |= after_ids == i
    roi = F.max_pool2d(support.float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
    inside, outside = delta[roi], delta[~roi]
    total = delta.sum()
    return {'feature_change_mse': float(delta.mean()),
            'change_energy_inside_support_fraction': float(delta[roi].sum() / total) if total > 0 else None,
            'support_fraction': float(roi.float().mean()),
            'inside_change_mse': float(inside.mean()) if len(inside) else None,
            'outside_change_mse': float(outside.mean()) if len(outside) else None,
            'exactly_unchanged': bool(torch.equal(before, after)),
            'support_definition': 'union of edited-slot argmax regions before/after; dilated one patch',
            'independent_ground_truth': False}, delta, roi
