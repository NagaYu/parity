"""Multi-tokenizer serving: correct output, and a prefix cache that stays valid.

The two things worth testing here are the two things that usually go wrong when
one model serves several tokenizers:

1. **Cross-contamination.**  A request on one view must produce exactly what it
   would have produced alone, even when batched next to requests on other views.
2. **Cache corruption.**  A KV prefix filled by one view must be numerically
   valid for another — Parity's whole efficiency argument is that views share an
   id space, so a cache entry means the same thing everywhere.
"""

from __future__ import annotations

import torch

from parity.serving import MultiTokenizerRouter, Request
from parity.serving.prefix_cache import PrefixCache, clone_kv, crop_kv, kv_length


def test_views_are_registered_and_masks_are_disjoint(router, attached, sample):
    sample_ja_text = sample.by_lang["ja"][0]
    V = attached["tokenizer"].base_vocab_size
    total = attached["tokenizer"].total_vocab_size
    # Each pack gets both a full view and a read-only ("lang:in") view.
    assert router.views() == ["base", "ja", "ja:in"]

    base_mask = router.logit_mask("base")
    ja_mask = router.logit_mask("ja")
    assert base_mask.shape == (total,)
    assert torch.isinf(base_mask[V:]).all()
    assert not torch.isinf(base_mask[:V]).any()
    assert not torch.isinf(ja_mask).any(), "the ja view should reach every id"

    # The read-only view reads pack ids but must never emit one. On a tied model
    # this is what makes emission drift zero rather than merely bounded.
    in_mask = router.logit_mask("ja:in")
    assert torch.isinf(in_mask[V:]).all()
    assert not torch.isinf(in_mask[:V]).any()
    assert router.encode(sample_ja_text, "ja:in") == router.encode(sample_ja_text, "ja")


def test_batched_generation_matches_isolated_generation(router, sample):
    """Mixed-view batching must not change any single request's output."""
    requests = [
        Request(sample.by_lang["en"][0], "base", 8),
        Request(sample.by_lang["ja"][1], "ja", 8),
        Request(sample.by_lang["en"][2], "base", 8),
        Request(sample.by_lang["ja"][3], "ja", 8),
        Request(sample.by_lang["ja"][4], "ja", 8),
    ]
    alone = [router.generate(r, use_cache=False) for r in requests]
    together = router.batch_generate(requests)
    for a, b, r in zip(alone, together, requests):
        assert a.prompt_ids == b.prompt_ids
        assert a.output_ids == b.output_ids, f"view {r.view} changed when batched with other views"
        assert a.view == b.view


def test_generation_never_emits_an_out_of_view_token(router, sample):
    V = router.tok.base_vocab_size
    for text in sample.by_lang["en"][:4]:
        out = router.generate(Request(text, "base", 12), use_cache=False)
        assert all(i < V for i in out.output_ids), "the base view emitted a pack token"


def test_decode_is_view_independent(router, attached, sample):
    aug = attached["tokenizer"]
    for text in sample.by_lang["ja"]:
        ja_ids = router.encode(text, "ja")
        base_ids = router.encode(text, "base")
        assert router.decode(ja_ids) == router.decode(base_ids)


def test_prefix_cache_hit_does_not_change_the_result(router, sample):
    router.cache.clear()
    req = Request(sample.by_lang["ja"][1], "ja", 6)
    cold = router.generate(req, use_cache=True)
    warm = router.generate(req, use_cache=True)
    assert cold.output_ids == warm.output_ids
    assert warm.cached_prefix_tokens > 0, "the second identical request should have hit the cache"
    assert router.cache.stats.hit_rate > 0


def test_prefix_cache_entry_is_valid_across_views(router, attached, sample):
    """The load-bearing claim: an id prefix means the same thing in every view.

    We fill the cache from a *base*-view request and then serve a *ja*-view
    request whose id prefix matches, and require the result to equal a cold run.
    """
    aug = attached["tokenizer"]
    router.cache.clear()

    # A Japanese string whose ja-view encoding shares a genuine id prefix with
    # some base-view encoding: use the ja-view ids themselves as the request and
    # seed the cache under the base label.
    text = sample.by_lang["ja"][1]
    ids = router.encode(text, "ja")
    assert len(ids) > 4

    cold = router.generate(Request(text, "ja", 6), use_cache=False)

    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=router.lm.device)
    _, kv = router.lm.run(inp, use_cache=True)
    router.cache.insert(ids[:-1], kv, view="base")  # deliberately mislabelled

    warm = router.generate(Request(text, "ja", 6), use_cache=True)
    assert warm.cached_prefix_tokens == len(ids) - 1
    assert router.cache.stats.cross_view_hits == 1, "the hit should be recorded as crossing views"
    assert warm.output_ids == cold.output_ids, "a cross-view cache hit changed the output"


def test_prefix_cache_longest_prefix_and_eviction():
    cache = PrefixCache(max_entries=2, min_prefix=2)
    cache.insert([1, 2, 3], kv="a")
    cache.insert([1, 2, 3, 4, 5], kv="b")
    entry, n = cache.lookup([1, 2, 3, 4, 5, 6])
    assert n == 5 and entry.kv == "b"
    entry, n = cache.lookup([1, 2, 3, 9])
    assert n == 3 and entry.kv == "a"
    assert cache.lookup([9, 9, 9] )[0] is None
    cache.insert([7, 8, 9], kv="c")  # forces an eviction
    assert len(cache) == 2
    assert cache.stats.evictions == 1


def test_kv_helpers_round_trip(router, sample):
    ids = router.encode(sample.by_lang["ja"][0], "ja")
    inp = torch.tensor([ids], dtype=torch.long, device=router.lm.device)
    _, kv = router.lm.run(inp, use_cache=True)
    assert kv_length(kv) == len(ids)
    cloned = clone_kv(kv)
    assert kv_length(cloned) == len(ids)
    cropped = crop_kv(cloned, 3)
    assert kv_length(cropped) == 3
    assert kv_length(kv) == len(ids), "cropping a clone must not disturb the original"


def test_throughput_comparison_reports_finite_numbers(router, sample):
    requests = [Request(sample.by_lang["ja"][i % 8], "ja", 4) for i in range(4)] + [
        Request(sample.by_lang["en"][i % 8], "base", 4) for i in range(4)
    ]
    cmp = router.compare_single_vs_multi(requests)
    assert cmp["multi_view"]["total_tokens"] < cmp["base_only"]["total_tokens"], (
        "the pack should reduce the number of tokens the same workload costs"
    )
    assert 0.0 <= cmp["dispatch_overhead"] < 1.0
    assert cmp["multi_view"]["tokens_per_second"] > 0


def test_router_refuses_a_model_tokenizer_size_mismatch(attached, base_tokenizer):
    from parity.adapters import TorchLMAdapter
    from parity.tiny import build_tiny_model
    import pytest

    fresh = TorchLMAdapter(build_tiny_model(base_tokenizer.vocab_size, seed=0, tie=False), name="tiny")
    with pytest.raises(ValueError, match="embedding rows"):
        MultiTokenizerRouter(fresh, attached["tokenizer"])
