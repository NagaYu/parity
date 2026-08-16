"""The reduction claim: fertility falls, by at least the target, on held-out text.

Also checks the measurement itself is honest — that ``tokens_per_word`` is
refused for scripts without word spaces rather than silently computed from
whitespace, which would produce a number that looks comparable and is not.
"""

from __future__ import annotations

import pytest

from parity.corpora import count_chars, count_words
from parity.fertility import measure_fertility, summarize_reduction
from parity.tokenization import BASE_VIEW

#: The reduction the fixture build must achieve on held-out Japanese text.
#: Deliberately modest: the fixture is a 900-token BPE and a 48-row budget.
#: Real builds on real tokenizers are expected far above this, and
#: ``benchmarks/run.py`` reports the real number.
TARGET_REDUCTION = 0.05


def test_tokens_per_word_is_refused_for_unsegmented_scripts():
    assert count_words("The children are playing.", "en") == 4
    assert count_words("बच्चे पार्क में खेल रहे हैं।", "hi") == 6
    assert count_words("子どもたちが公園で遊んでいます。", "ja") is None
    assert count_words("เด็กกำลังเล่น", "th") is None


def test_char_count_excludes_whitespace_and_normalises():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "が")
    assert len(nfd) == 2 and count_chars(nfd) == 1
    assert count_chars("a b  c") == 3


def test_baseline_shows_the_asymmetry_the_project_exists_to_repair(plain_augmented, sample):
    """A tokenizer fit mostly on English charges other scripts more per meaning."""
    aug = plain_augmented
    ratios = {}
    for lang in ("ja", "hi", "ar", "th", "sw"):
        rep = measure_fertility(aug, sample.by_lang[lang], lang, BASE_VIEW, sample.by_lang["en"])
        ratios[lang] = rep.parity_ratio
        assert rep.parity_ratio > 1.0
        assert 0.0 < rep.effective_context_fraction < 1.0
    en = measure_fertility(aug, sample.by_lang["en"], "en", BASE_VIEW, sample.by_lang["en"])
    assert abs(en.parity_ratio - 1.0) < 1e-9


def test_pack_reduces_tokens_on_held_out_text(router, attached, sample):
    """The headline reduction, measured on sentences no split ever saw."""
    aug = attached["tokenizer"]
    base = sum(router.count(s, "base") for s in sample.by_lang["ja"])
    with_pack = sum(router.count(s, "ja") for s in sample.by_lang["ja"])
    assert with_pack < base
    reduction = 1 - with_pack / base
    assert reduction >= TARGET_REDUCTION, f"only {100 * reduction:.1f}% reduction, wanted >= {100 * TARGET_REDUCTION:.0f}%"


def test_reduction_summary_is_self_consistent(attached, sample):
    from parity.corpora import ParallelCorpus

    aug = attached["tokenizer"]
    corpus = ParallelCorpus(by_lang={"ja": sample.by_lang["ja"], "en": sample.by_lang["en"]}, source="sample")
    s = summarize_reduction(aug, corpus, "ja", aug.view("ja"))
    assert s.tokens_after < s.tokens_before
    assert 0 < s.token_reduction < 1
    assert s.context_gain == pytest.approx(1 / (1 - s.token_reduction), rel=1e-6)
    assert s.cost_reduction == pytest.approx(s.token_reduction)
    assert s.parity_ratio_after < s.parity_ratio_before
    assert s.tokens_per_new_embedding > 0


def test_pack_never_makes_any_string_longer(attached, sample):
    aug = attached["tokenizer"]
    view = aug.view("ja")
    for lang in sample.by_lang:
        for text in sample.by_lang[lang]:
            assert len(aug.encode(text, view)) <= len(aug.encode(text, BASE_VIEW))
