"""Target-free source program transport and region-based merge."""
from dataclasses import dataclass

import torch

from urbangene.budget import pack_program, unpack_program
from .models.composition.geometry import ProgramGeometry


def wire_roundtrip(program, cfg):
    device = program["gene_ids"].device
    restored = [unpack_program(pack_program(program, cfg, "urbangene", i), cfg, "urbangene")
                for i in range(program["gene_ids"].shape[0])]
    return {k: torch.cat([p[k] for p in restored]).to(device) for k in program}


@dataclass
class MergeResult:
    program: dict | None
    overflow: bool
    requested_instances: int
    selected_a: list
    selected_b: list


def merge_programs(program_a, program_b, region, cfg, overlap_threshold=0.5):
    """One scene at a time. Overlap=sampled support weight inside Ω / full support.

    Instances in A with overlap >= threshold are removed; corresponding B
    instances are inserted. No slot-index matching or target information.
    """
    assert program_a["gene_ids"].shape[0] == program_b["gene_ids"].shape[0] == 1
    assert 0 < overlap_threshold <= 1
    geometry = ProgramGeometry(cfg).to(program_a["gene_ids"].device)
    x0, y0, x1, y1 = region
    assert 0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256
    coords = (geometry.canvas_coords+1)*128
    region_mask = ((coords[:, 0] >= x0) & (coords[:, 0] < x1) &
                   (coords[:, 1] >= y0) & (coords[:, 1] < y1))
    # Geometry requires template shape, but values below are constant and do not
    # depend on a learned library; only program positions/scales determine support.
    class OnesLibrary:
        def lookup(self, ids):
            m = cfg["model"]
            return torch.ones(*ids.shape, m["latent_channels"], m["gene_patch_hw"], m["gene_patch_hw"], device=ids.device)
    selected = []
    for program, donor in [(program_a, False), (program_b, True)]:
        sample = geometry.sample(geometry(OnesLibrary(), program))
        alpha = sample["alpha"][0]
        fraction = alpha[region_mask].sum(0)/alpha.sum(0).clamp_min(1e-8)
        inside = fraction >= overlap_threshold
        select = program["enabled"][0].bool() & (inside if donor else ~inside)
        selected.append(select.nonzero().flatten().tolist())
    n = sum(map(len, selected)); capacity = cfg["model"]["max_instances"]
    if n > capacity:
        return MergeResult(None, True, n, *selected)
    merged = {k: torch.zeros_like(v) for k, v in program_a.items()}
    for field in merged:
        values = torch.cat([program_a[field][0, selected[0]], program_b[field][0, selected[1]]])
        merged[field][0, :n] = values
    merged = wire_roundtrip(merged, cfg)
    return MergeResult(merged, False, n, *selected)


@torch.no_grad()
def predict_recombination(model, source_a, source_b, region, threshold=0.5):
    """NO target / scene_spec / target mask argument."""
    model.eval()
    z = model.ae.encoder(torch.cat([source_a, source_b], 0))
    program, _ = model.parse_latent(z)
    transported = wire_roundtrip(program, model.cfg)
    pa = {k: v[:1] for k, v in transported.items()}; pb = {k: v[1:2] for k, v in transported.items()}
    merged = merge_programs(pa, pb, region, model.cfg, threshold)
    if merged.overflow:
        return dict(merge=merged, prediction=None, latent_hat=None)
    output = model.compose_program(merged.program)
    return dict(merge=merged, prediction=output["image_hat"], latent_hat=output["latent_hat"])
