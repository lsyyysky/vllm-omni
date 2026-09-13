"""Stage input processor for MammothModa2 (AR -> DiT)."""

from collections.abc import Mapping
from typing import Any


def _as_dict(prompt: Any) -> dict[str, Any]:
    """Coerce an original-stage prompt to a dict.

    It may arrive as a dict, a NamedTuple/object, or a bare string depending on
    the calling flow (the shared text_to_image example vs the bespoke script).
    """
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _coerce_dim(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return iv if iv > 0 else default


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Any | None = None,
    _requires_multimodal_data: bool = False,
) -> dict[str, Any]:
    """Convert MammothModa2 AR output into a diffusion request prompt."""
    if not source_outputs:
        raise ValueError("MammothModa2 AR stage produced no outputs")

    ar_output = source_outputs[0]
    prompt_dict = _as_dict(prompt)
    addi_info = prompt_dict.get("additional_information") or {}
    mm_kwargs = prompt_dict.get("mm_processor_kwargs") or {}

    image_height = _coerce_dim(
        mm_kwargs.get("target_h"),
        _coerce_dim((addi_info.get("image_height") or [None])[0], 1024),
    )
    image_width = _coerce_dim(
        mm_kwargs.get("target_w"),
        _coerce_dim((addi_info.get("image_width") or [None])[0], 1024),
    )

    prompt_token_ids = ar_output.prompt_token_ids
    completion_output = ar_output.outputs[0]
    # The final generated token has no corresponding hidden state.
    gen_token_ids = completion_output.cumulative_token_ids[:-1]
    full_token_ids = prompt_token_ids + gen_token_ids

    mm_output = getattr(completion_output, "multimodal_output", None)
    if not isinstance(mm_output, Mapping) or "latent" not in mm_output:
        raise ValueError(
            "AR stage output missing latent multimodal output. "
            f"request_id={getattr(ar_output, 'request_id', None)}, "
            f"completion_has_mm={hasattr(completion_output, 'multimodal_output')}"
        )
    full_hidden_states = mm_output["latent"]
    hidden_total = int(full_hidden_states.shape[0])
    expected_hidden_total = len(prompt_token_ids) + len(gen_token_ids)
    if hidden_total != expected_hidden_total:
        raise ValueError(
            f"Hidden states length mismatch: expected {expected_hidden_total}, got {hidden_total}"
        )

    return {
        "prompt": "",
        "height": image_height,
        "width": image_width,
        "extra": {
            # The serializer has no bf16 representation; the DiT casts back to
            # its execution dtype after receiving the cross-stage payload.
            "full_hidden_states": full_hidden_states.float().contiguous(),
            "full_token_ids": full_token_ids,
            "answer_start_index": len(prompt_token_ids),
        },
    }
