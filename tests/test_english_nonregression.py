"""The non-regression claim: English is not merely preserved, it is untouched.

Parity's design makes this an identity rather than an empirical result:

* packs only *append* embedding rows, so every pre-existing row is bit-identical;
* the base view's tokenizer cannot emit a pack id;
* the base view's logit mask sets every pack logit to −∞ before the softmax, and
  masked-softmax over the base subset equals the original softmax exactly.

The tests below check each link, and then the composition, numerically — because
an argument that is correct on paper and wrong in the code is still wrong.
"""

from __future__ import annotations

import pytest
import torch

from parity.serving import MultiTokenizerRouter, Request


def _base_logits(adapter, ids):
    with torch.no_grad():
        return adapter.model(input_ids=torch.tensor([list(ids)])).logits[0].float().clone()


def test_appending_rows_does_not_touch_existing_rows(adapter, built):
    pack = built["result"].pack
    before = adapter.input_matrix().detach().clone()
    v0 = adapter.vocab_size()
    adapter.append_rows(pack.input_embeddings, pack.output_embeddings)
    after = adapter.input_matrix()[:v0].detach()
    assert torch.equal(before, after)
    assert adapter.vocab_size() == v0 + len(pack)


def test_base_view_logits_are_bit_identical(base_tokenizer, built, sample):
    """The strong form: not 'close', identical."""
    from parity.build import attach_and_verify
    from parity.tiny import build_tiny_model
    from parity.adapters import TorchLMAdapter

    adapter = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    probes = sample.by_lang["en"]
    reference = [_base_logits(adapter, base_tokenizer.encode(t)) for t in probes]

    aug = attach_and_verify(adapter, base_tokenizer, [built["result"].pack])
    router = MultiTokenizerRouter(adapter, aug)

    for text, ref in zip(probes, reference):
        ids = router.encode(text, "base")
        assert ids == base_tokenizer.encode(text)
        got = router.logits(ids, "base")
        assert torch.equal(got[: aug.base_vocab_size], ref[-1]), "base-view logits moved"
        assert torch.isinf(got[aug.base_vocab_size :]).all(), "a pack token was reachable from the base view"


def test_base_view_distribution_equals_the_original_softmax(router, attached, sample, base_tokenizer):
    """Masking then renormalising must reproduce the original distribution."""
    adapter = attached["adapter"]
    V = attached["tokenizer"].base_vocab_size
    for text in sample.by_lang["en"][:6]:
        ids = base_tokenizer.encode(text)
        with torch.no_grad():
            full = adapter.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
        original = torch.softmax(full[:V], dim=-1)
        masked = torch.softmax(router.logits(ids, "base"), dim=-1)
        assert torch.allclose(masked[:V], original, atol=0, rtol=0)
        assert float(masked[V:].sum()) == 0.0


def test_english_generation_is_unchanged_by_a_japanese_pack(base_tokenizer, built, sample):
    from parity.adapters import TorchLMAdapter
    from parity.build import attach_and_verify
    from parity.tiny import build_tiny_model
    from parity.tokenization import AugmentedTokenizer

    clean = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    clean_router = MultiTokenizerRouter(clean, AugmentedTokenizer(base_tokenizer))

    packed = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    aug = attach_and_verify(packed, base_tokenizer, [built["result"].pack])
    packed_router = MultiTokenizerRouter(packed, aug)

    for text in sample.by_lang["en"][:6]:
        a = clean_router.generate(Request(text, "base", 8), use_cache=False)
        b = packed_router.generate(Request(text, "base", 8), use_cache=False)
        assert a.prompt_ids == b.prompt_ids
        assert a.output_ids == b.output_ids, "English generation changed after attaching a Japanese pack"


def test_english_token_count_is_unchanged(router, sample):
    for text in sample.by_lang["en"]:
        assert router.count(text, "base") == router.count(text, "ja"), (
            "a Japanese pack changed the token count of English text"
        )


def test_english_bits_per_character_is_unchanged_on_the_base_view(base_tokenizer, built, sample):
    """Non-regression on the metric a benchmark would actually report.

    The claim is about the **base view**: an English request served with the
    original tokenizer is served by the original model. So we compare a clean
    model against a packed one, both on the base view — not base-view against
    pack-view, which is a different question answered by the test below.
    """
    from benchmarks.tasks import bits_per_character
    from parity.adapters import TorchLMAdapter
    from parity.build import attach_and_verify
    from parity.tiny import build_tiny_model
    from parity.tokenization import AugmentedTokenizer

    clean = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    clean_router = MultiTokenizerRouter(clean, AugmentedTokenizer(base_tokenizer))

    packed = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    aug = attach_and_verify(packed, base_tokenizer, [built["result"].pack])
    packed_router = MultiTokenizerRouter(packed, aug)

    before = bits_per_character(clean_router, sample.by_lang["en"], "base")
    after = bits_per_character(packed_router, sample.by_lang["en"], "base")
    assert after == pytest.approx(before, abs=1e-9)


def test_serving_english_under_a_pack_view_costs_only_the_masked_mass(router, attached, sample):
    """An operator who serves English *through* a pack view pays a tiny, bounded price.

    This is the honest edge of the non-regression claim. Under the ``ja`` view
    the softmax denominator also spans the Japanese pack rows, so English
    log-probabilities shift by exactly the mass those rows take. Token counts do
    not change at all, and the shift is reported rather than hidden — the
    zero-cost path is to route English to the base view, which is what
    :class:`MultiTokenizerRouter` does by default.
    """
    from benchmarks.tasks import bits_per_character

    base = bits_per_character(router, sample.by_lang["en"], "base")
    via_pack = bits_per_character(router, sample.by_lang["en"], "ja")
    assert via_pack >= base - 1e-9, "masking more tokens cannot increase probability"
    assert via_pack - base < 0.5, f"pack rows absorbed {via_pack - base:.3f} bits/char of English mass"
    for text in sample.by_lang["en"]:
        assert router.count(text, "base") == router.count(text, "ja")
