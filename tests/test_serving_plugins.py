"""The vLLM and TGI adapters, exercised without vLLM and without a server.

Both adapters are thin by design — they delegate every view decision to
:class:`~parity.serving.multi_tokenizer.MultiTokenizerRouter`. These tests pin
that down: the adapters must produce the *same* ids and the *same* allowed-token
set as the router, or the safety property stops being one property and becomes
three.
"""

from __future__ import annotations

import json

import pytest
import torch

from serving.tgi_plugin import TGISidecar
from serving.vllm_plugin import ParityTokenizerRegistry, ViewLogitsProcessor, vllm_available


def test_vllm_availability_probe_does_not_raise():
    assert isinstance(vllm_available(), bool)


def test_registry_encoding_matches_the_router(router, sample):
    reg = ParityTokenizerRegistry(router)
    assert reg.views() == router.views()
    for text in sample.by_lang["ja"][:6]:
        assert reg.encode(text, "ja") == router.encode(text, "ja")
        assert reg.encode(text, "base") == router.encode(text, "base")
        assert reg.decode(reg.encode(text, "ja")) == text


def test_view_logits_processor_masks_exactly_the_router_mask(router, attached):
    reg = ParityTokenizerRegistry(router)
    V = attached["tokenizer"].base_vocab_size
    total = attached["tokenizer"].total_vocab_size

    proc = reg.logits_processor("base")
    assert isinstance(proc, ViewLogitsProcessor)
    logits = torch.zeros(total)
    out = proc([1, 2, 3], logits)
    assert torch.isinf(out[V:]).all(), "the base view must not be able to emit a pack token"
    assert not torch.isinf(out[:V]).any()

    out_ja = reg.logits_processor("ja")([1], torch.zeros(total))
    assert not torch.isinf(out_ja).any()


def test_tgi_sidecar_request_carries_ids_and_ban_list(router, sample, attached):
    side = TGISidecar(router, dry_run=True)
    V = attached["tokenizer"].base_vocab_size
    text = sample.by_lang["ja"][1]

    body = side.build_request(text, "ja", max_new_tokens=8)
    assert body["parameters"]["parity_input_ids"] == router.encode(text, "ja")
    assert body["parameters"]["parity_banned_token_ids"] == []

    base_body = side.build_request(text, "base", max_new_tokens=8)
    banned = set(base_body["parameters"]["parity_banned_token_ids"])
    assert banned == set(range(V, attached["tokenizer"].total_vocab_size))
    assert len(base_body["parameters"]["parity_input_ids"]) >= len(body["parameters"]["parity_input_ids"])
    json.dumps(body)  # must be serialisable as-is


def test_tgi_sidecar_reports_the_saving(router, sample):
    side = TGISidecar(router, dry_run=True)
    saving = side.token_savings(sample.by_lang["ja"][1], "ja")
    assert saving["view_tokens"] <= saving["base_tokens"]
    assert saving["saved"] == saving["base_tokens"] - saving["view_tokens"]
    assert 0.0 <= saving["reduction"] < 1.0
    # English through a Japanese pack view must cost exactly the same.
    en = side.token_savings(sample.by_lang["en"][0], "ja")
    assert en["saved"] == 0


def test_tgi_dry_run_generate_is_offline(router, sample):
    side = TGISidecar(router, tgi_url="http://127.0.0.1:1", dry_run=True)
    out = side.generate(sample.by_lang["ja"][2], "ja", max_new_tokens=4)
    assert out["dry_run"] and out["prompt_tokens"] > 0


def test_vllm_unwrap_reports_a_useful_error_when_internals_move():
    from serving.vllm_plugin import _unwrap_model

    class NotVLLM:
        pass

    with pytest.raises(RuntimeError, match="vLLM's internals have moved"):
        _unwrap_model(NotVLLM())
