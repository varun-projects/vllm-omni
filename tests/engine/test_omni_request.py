# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequest substitutability with the base Request.

vllm-omni rebinds ``vllm.v1.request.Request`` to ``OmniRequest`` at import
time (see ``vllm_omni/patch.py``). vLLM core constructs ``Request`` positionally
in some paths (notably ``vllm/v1/worker/gpu/warmup.py::warmup_kernels`` for V2
model-runner architectures like Qwen3ForCausalLM). The omni-specific params
must therefore be keyword-only so positional construction stays Liskov-
substitutable with the base class — including base-style calls that pass
``prompt_embeds`` (itself a positional-capable base param) positionally.
"""

import inspect

import msgspec
import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest, PromptEmbedsPayload
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_params_are_keyword_only():
    """The three omni params must be keyword-only after ``*args``.

    Guards against re-introducing the import-time rebind bug where omni params
    came first positionally and broke positional ``Request(...)`` construction.
    """
    params = inspect.signature(OmniRequest.__init__).parameters
    for name in (
        "prompt_embeds",
        "external_req_id",
        "additional_information",
        "model_intermediate_buffer",
    ):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_positional_construction_matches_base_request():
    """Reproduces the warmup_kernels call: Request(id, tokens, sp, pp)."""
    req = OmniRequest("req-0", [1, 2, 3], SamplingParams(), None)

    assert isinstance(req, Request)
    assert req.request_id == "req-0"
    assert req.prompt_token_ids == [1, 2, 3]
    # Omni params default cleanly when constructed positionally.
    assert req.external_req_id is None
    assert req.additional_information is None
    assert req.prompt_embeds_payload is None
    assert req.num_stale_output_tokens == 0


def test_positional_prompt_embeds_does_not_collide():
    """Base-style positional call that includes ``prompt_embeds``.

    ``prompt_embeds`` is a positional-capable base param (after
    ``arrival_time``). A base-style call must not trip
    ``got multiple values for argument 'prompt_embeds'`` — the subclass must
    not also inject its keyword override when the value arrived positionally.
    Positional order: request_id, prompt_token_ids, sampling_params,
    pooling_params, client_index, arrival_time, prompt_embeds.
    """
    embeds = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    req = OmniRequest("req-pe", [1, 2], SamplingParams(), None, 0, None, embeds)

    assert isinstance(req, Request)
    assert torch.equal(req.prompt_embeds, embeds)
    # Positional tensor is not a serialized payload.
    assert req.prompt_embeds_payload is None
    assert req.external_req_id is None


def test_keyword_omni_params_round_trip():
    """Keyword omni params are preserved; serialized embeds are decoded."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = PromptEmbedsPayload(data=arr.tobytes(), shape=[2, 3], dtype="float32")

    req = OmniRequest(
        request_id="req-1",
        prompt_token_ids=[7, 8],
        sampling_params=SamplingParams(),
        pooling_params=None,
        prompt_embeds=payload,
        external_req_id="ext-1",
    )

    assert req.external_req_id == "ext-1"
    # The serialized payload is retained and decoded into a tensor on the base.
    assert req.prompt_embeds_payload is payload
    assert torch.equal(req.prompt_embeds, torch.from_numpy(arr))


def test_model_intermediate_buffer_round_trips_to_streaming_update():
    buffer = {"duplex": {"turn_id": 3}}
    request = OmniRequest(
        request_id="req-model-buffer",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        resumable=True,
        model_intermediate_buffer=buffer,
    )

    update = OmniStreamingUpdate.from_request(request)

    assert update is not None
    assert request.model_intermediate_buffer is buffer
    assert update.model_intermediate_buffer is buffer


def test_kv_hints_survive_omni_upgrade_wire_transport_and_scheduler_conversion():
    hints = KvHintsEnvelope("1", "hint-message", [KvHintAction("a", "prefetch", "1", {"blocks": [7]})])
    core = EngineCoreRequest(
        request_id="hint-request",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session",
        kv_hints=hints,
    )
    upgraded = OmniEngineCoreRequest.from_request(core)
    # v0.34 wires prompt token ids as a packed int32 ndarray (the plain-list
    # decode raises "Expected `numpy.ndarray`, got `list`"); the real engine
    # core transport packs via `_with_packed_prompt` and serializes with vLLM's
    # MsgpackEncoder/MsgpackDecoder pair.
    packed = _with_packed_prompt(upgraded)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(packed))
    scheduled = OmniRequest.from_engine_core_request(decoded, block_hasher=None)
    assert decoded.session_id == "session"
    assert scheduled.kv_hints == hints
    assert scheduled.session_id == "session"


# vllm_omni/patch.py rebinds ``vllm.v1.engine.EngineCoreRequest`` to the Omni
# subclass, so take the unpatched upstream class from the subclass's base.
_UPSTREAM_ENGINE_CORE_REQUEST = OmniEngineCoreRequest.__bases__[0]

# ``from_request`` keyword overrides; a passed value replaces the source field.
_FROM_REQUEST_OVERRIDES = (
    "prompt_embeds",
    "additional_information",
    "model_intermediate_buffer",
    "payload_sender_info",
)


class _FieldSentinel:
    """Unique per-field marker; msgspec Struct construction does not type-check."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<sentinel {self.name}>"


def _dropped_fields(source, result, skip=()) -> list[str]:
    return [
        f.name
        for f in msgspec.structs.fields(type(source))
        if f.name not in skip and getattr(result, f.name, None) is not getattr(source, f.name)
    ]


@pytest.mark.parametrize("source_cls", [_UPSTREAM_ENGINE_CORE_REQUEST, OmniEngineCoreRequest], ids=["upstream", "omni"])
def test_from_request_preserves_every_struct_field(source_cls):
    """Guard against vLLM adding an ``EngineCoreRequest`` field that ``from_request`` drops (#4015).

    Every field gets a distinct sentinel, so a new field with a default is
    still caught. Fix a failure by copying the named field in
    ``OmniEngineCoreRequest.from_request``.
    """
    assert _UPSTREAM_ENGINE_CORE_REQUEST.__module__ == "vllm.v1.engine"
    field_names = [f.name for f in msgspec.structs.fields(source_cls)]
    source = source_cls(**{name: _FieldSentinel(name) for name in field_names})

    upgraded = OmniEngineCoreRequest.from_request(source)

    assert type(upgraded) is OmniEngineCoreRequest
    dropped = _dropped_fields(source, upgraded)
    assert not dropped, f"OmniEngineCoreRequest.from_request drops {source_cls.__name__} fields: {dropped}"

    overrides = {name: _FieldSentinel(f"override-{name}") for name in _FROM_REQUEST_OVERRIDES}
    overridden = OmniEngineCoreRequest.from_request(source, **overrides)

    for name, value in overrides.items():
        assert getattr(overridden, name) is value, name
    dropped = _dropped_fields(source, overridden, skip=_FROM_REQUEST_OVERRIDES)
    assert not dropped, f"OmniEngineCoreRequest.from_request drops {source_cls.__name__} fields: {dropped}"
