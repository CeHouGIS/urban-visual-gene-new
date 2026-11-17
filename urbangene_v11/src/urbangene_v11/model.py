import torch
import torch.nn as nn

from urbangene.budget import layout
from urbangene.models.gene_library import GeneLibrary
from urbangene.models.parser import SceneParser
from .models.composition import GRCTComposer, FixedAverage, FixedAlpha, ResidualCNN


class UrbanGeneV11(nn.Module):
    def __init__(self, cfg, ae):
        super().__init__(); self.cfg = cfg
        m = cfg["model"]; self.budget = layout(cfg, "urbangene")
        assert self.budget["bits"] == 272, "V1.1 uses exactly the V0 272-bit interface"
        self.ae = ae.freeze()
        self.parser = SceneParser(cfg, "urbangene", self.budget)
        target = m.get("parser_parameter_target")
        if target is not None:
            # R0-P: all added parameters participate in the actual parser.
            width = m["parser_width"]; outputs = self.parser.count*self.parser.size
            fixed = sum(p.numel() for p in self.parser.features.parameters())
            hidden = max(8, round((target-fixed-outputs)/(width*64+outputs+1)))
            self.parser.context[1] = nn.Linear(width*64, hidden)
            self.parser.head = nn.Linear(hidden, outputs)
        self.library = GeneLibrary(m["gene_count"], m["latent_channels"], m["gene_patch_hw"])
        kind = m["composer"]["type"]
        self.composer = {"fixed_average": FixedAverage, "fixed_alpha": FixedAlpha,
                         "residual_cnn": ResidualCNN, "grct": GRCTComposer}[kind](cfg)

    def train(self, mode=True):
        super().train(mode); self.ae.eval()
        return self

    def parse_latent(self, latent, temperature=1.):
        program, st = self.parser(latent, temperature)
        return self._ensure_nonempty(program, st)

    @staticmethod
    def _ensure_nonempty(program, st):
        enabled = program["enabled"].bool()
        empty = ~enabled.any(-1)
        if empty.any():
            # Non-empty scenes must not fall into a zero-output, zero-gradient
            # basin. Pick the fallback only from serialized instance fields.
            code = ((program["gene_ids"]*256+program["position_ids"])*3+
                    program["scale_ids"])
            fallback = code.argmin(-1)
            enabled = enabled.clone()
            enabled[empty, fallback[empty]] = True
            program["enabled"] = enabled.long()
            hard = enabled.to(st["enabled"].dtype)
            st["enabled"] = hard+st["enabled"]-st["enabled"].detach()
        return program, st

    def forward_latent(self, latent, temperature=1., decode=False):
        program, st = self.parse_latent(latent, temperature)
        z = self.composer(self.library, program, st if self.training else None)
        return dict(latent_hat=z, image_hat=self.ae.decoder(z) if decode else None, program=program)

    def forward(self, images):
        with torch.no_grad():
            z = self.ae.encoder(images)
        return self.forward_latent(z, decode=True)

    def compose_program(self, program, decode=True):
        z = self.composer(self.library, program)
        return dict(latent_hat=z, image_hat=self.ae.decoder(z) if decode else None, program=program)

    def parameter_counts(self):
        return {name: sum(p.numel() for p in getattr(self, name).parameters())
                for name in ["ae", "parser", "library", "composer"]}
