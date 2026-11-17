from pathlib import Path

import pytest
import torch

from urbangene.utils import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    c["data"]["root"] = str(tmp_path / "data")
    c["data"]["counts"] = {k: 8 for k in c["data"]["counts"]}
    c["ae"]["epochs"] = 1; c["training"]["epochs"] = 1
    torch.set_num_threads(2)
    return c
