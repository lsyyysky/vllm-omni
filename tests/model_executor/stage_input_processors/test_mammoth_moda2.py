# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the MammothModa2 AR-to-diffusion bridge."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import (
    ar2diffusion,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_output(multimodal_output: dict) -> SimpleNamespace:
    completion = SimpleNamespace(
        cumulative_token_ids=[152072, 152073, 152064],
        multimodal_output=multimodal_output,
    )
    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[10, 11],
        outputs=[completion],
    )


def test_ar2diffusion_builds_hidden_state_payload() -> None:
    hidden_states = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)

    result = ar2diffusion(
        [_ar_output({"latent": hidden_states})],
        {
            "prompt": "ignored after AR encoding",
            "mm_processor_kwargs": {"target_h": 512, "target_w": 768},
        },
    )

    assert result["prompt"] == ""
    assert result["height"] == 512
    assert result["width"] == 768
    assert result["extra"]["full_token_ids"] == [10, 11, 152072, 152073]
    assert result["extra"]["answer_start_index"] == 2
    assert result["extra"]["full_hidden_states"].dtype == torch.float32
    torch.testing.assert_close(result["extra"]["full_hidden_states"], hidden_states.float())


def test_ar2diffusion_rejects_missing_hidden_states() -> None:
    with pytest.raises(ValueError, match="missing latent"):
        ar2diffusion([_ar_output({})], {})
