import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.bagel.bagel_transformer import Qwen2MoTModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bagel_exposes_hsdp_shard_conditions_for_transformer_blocks():
    model = object.__new__(Qwen2MoTModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["layers.0", "layers.1"]
