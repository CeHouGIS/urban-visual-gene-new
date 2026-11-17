import pytest
import torch

from urbangene_v11.config import PROJECT, load_config


@pytest.fixture
def cfg():
    torch.set_num_threads(2); torch.manual_seed(42)
    return load_config(PROJECT / "configs/smoke.yaml")


@pytest.fixture
def program(cfg):
    n = cfg["model"]["max_instances"]
    return dict(gene_ids=torch.arange(n)[None] % 64,
                enabled=torch.tensor([[1]*8+[0]*8]),
                position_ids=torch.tensor([[85, 86, 101, 102, 153, 154, 169, 170]*2]),
                scale_ids=torch.arange(n)[None] % 3)
