"""Baseline measurement: what one language currently costs, in tokens.

This module is the *first* thing that runs and the last thing that is checked.
Everything Parity does is scored as a change to the numbers produced here.

The metric that matters
-----------------------
``parity_ratio`` = tokens(target sentence) / tokens(aligned English sentence),
on a parallel corpus.  It is the only cross-language token metric that is
simultaneously (a) meaningful for scripts without word spaces, (b) free of
normalisation artefacts, and (c) directly interpretable — a ratio of 2.4 means
this speaker gets 42% of the context window, and pays 2.4x per message, for the
same content.

Classical "fertility" (tokens per whitespace word) is reported too, but only
where whitespace words exist; see :func:`parity.corpora.count_words`.

A framing note that is part of the specification, not decoration: a high parity
ratio is a property of *the tokenizer*, produced by the corpus it was fit on.
It is not a property of the language, and nothing in this codebase should be
read or reported as saying otherwise.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from parity.corpora import ParallelCorpus, count_chars, count_words, normalize
from parity.tokenization import BASE_VIEW, AugmentedTokenizer, TokenizerView
from parity.types import FertilityReport

log = logging.getLogger("parity.fertility")


def measure_fertility(
    tok: AugmentedTokenizer,
    sentences: Sequence[str],
    lang: str,
    view: TokenizerView = BASE_VIEW,
    pivot_sentences: Optional[Sequence[str]] = None,
    pivot_view: TokenizerView = BASE_VIEW,
    tokenizer_id: str = "",
) -> FertilityReport:
    """Token cost of ``sentences`` under ``view``, optionally vs a pivot.

    ``pivot_sentences[i]`` must be a translation of ``sentences[i]``.  The pivot
    is always tokenized under ``pivot_view`` (the base view by default) because
    the comparison we want is "target language after Parity" against "English as
    it has always been", not against a hypothetically improved English.

    Claim: reduction — produces the before/after numbers that every headline
    percentage in the README is computed from.
    """
    total_tokens = 0
    total_chars = 0
    total_words: Optional[int] = 0
    per_sentence: List[int] = []
    ratios: List[float] = []

    for i, sent in enumerate(sentences):
        sent = normalize(sent)
        n_tok = tok.count(sent, view)
        per_sentence.append(n_tok)
        total_tokens += n_tok
        total_chars += count_chars(sent)
        w = count_words(sent, lang)
        if w is None:
            total_words = None
        elif total_words is not None:
            total_words += w
        if pivot_sentences is not None and i < len(pivot_sentences):
            n_piv = tok.count(normalize(pivot_sentences[i]), pivot_view)
            if n_piv > 0:
                ratios.append(n_tok / n_piv)

    return FertilityReport(
        lang=lang,
        tokenizer_id=tokenizer_id or getattr(tok.base, "name", "tokenizer"),
        n_sentences=len(sentences),
        total_tokens=total_tokens,
        total_chars=total_chars,
        total_words=total_words,
        # Corpus-level ratio (sum/sum) is the billing-relevant number; the median
        # of per-sentence ratios is reported alongside because a corpus ratio can
        # be dominated by a few long sentences.
        parity_ratio=(total_tokens / _pivot_total(tok, pivot_sentences, pivot_view)) if pivot_sentences else None,
        parity_ratio_median=(statistics.median(ratios) if ratios else None),
        per_sentence_tokens=per_sentence,
    )


def _pivot_total(tok: AugmentedTokenizer, pivot_sentences: Optional[Sequence[str]], view: TokenizerView) -> int:
    if not pivot_sentences:
        return 1
    return max(1, sum(tok.count(normalize(s), view) for s in pivot_sentences))


def fertility_table(
    tok: AugmentedTokenizer,
    corpus: ParallelCorpus,
    langs: Optional[Sequence[str]] = None,
    pivot: str = "en",
    view_for: Optional[Dict[str, TokenizerView]] = None,
    tokenizer_id: str = "",
) -> Dict[str, FertilityReport]:
    """Fertility for every language in ``corpus`` against the pivot.

    ``view_for`` lets the caller measure each language under its own pack while
    leaving the pivot on the base view — the exact configuration Parity ships.

    Claim: reduction — this table is the atlas row set and the benchmark's
    condition-A/D comparison.
    """
    langs = list(langs or corpus.langs())
    view_for = view_for or {}
    out: Dict[str, FertilityReport] = {}
    for lang in langs:
        sents = corpus.by_lang.get(lang)
        if not sents:
            continue
        # `pivot_for` keeps pairwise-aligned corpora (OPUS-100) honest: each
        # language is compared against *its own* English side, never another
        # language's.
        out[lang] = measure_fertility(
            tok,
            sents,
            lang,
            view=view_for.get(lang, BASE_VIEW),
            pivot_sentences=None if lang == pivot else corpus.pivot_for(lang, pivot),
            tokenizer_id=tokenizer_id,
        )
    return out


@dataclass
class ReductionSummary:
    """Before/after token cost for one language under one pack.

    Claim: reduction — the object the CLI prints and the benchmark aggregates.
    """

    lang: str
    tokens_before: int
    tokens_after: int
    parity_ratio_before: Optional[float]
    parity_ratio_after: Optional[float]
    n_sentences: int
    n_new_tokens: int

    @property
    def token_reduction(self) -> float:
        """Fraction of tokens removed on this corpus, in ``[0, 1)``.

        Claim: reduction — the x-axis of the Pareto figure.
        """
        if self.tokens_before <= 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before

    @property
    def context_gain(self) -> float:
        """Multiplicative increase in effective context, ``before/after``.

        A 30% token reduction is a 1.43x longer usable window — the more
        legible way to state the same fact.

        Claim: reduction — benchmark metric (5).
        """
        if self.tokens_after <= 0:
            return 1.0
        return self.tokens_before / self.tokens_after

    @property
    def cost_reduction(self) -> float:
        """Per-message inference cost saved, assuming linear per-token pricing.

        Equal to :attr:`token_reduction` for prompt tokens.  Stated separately
        because it is the number an operator budgets with.

        Claim: reduction, low-cost — benchmark metric (5).
        """
        return self.token_reduction

    @property
    def tokens_per_new_embedding(self) -> float:
        """Tokens saved per embedding row spent — the selection objective.

        Claim: reduction — the efficiency of the vocabulary budget, and the
        quantity the greedy submodular solver maximises.
        """
        if self.n_new_tokens <= 0:
            return 0.0
        return (self.tokens_before - self.tokens_after) / self.n_new_tokens

    def to_dict(self) -> Dict[str, float]:
        """Serialise, derived properties included.

        Claim: infrastructure.
        """
        return {
            "lang": self.lang,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "token_reduction": self.token_reduction,
            "context_gain": self.context_gain,
            "cost_reduction": self.cost_reduction,
            "parity_ratio_before": self.parity_ratio_before,
            "parity_ratio_after": self.parity_ratio_after,
            "n_sentences": self.n_sentences,
            "n_new_tokens": self.n_new_tokens,
            "tokens_per_new_embedding": self.tokens_per_new_embedding,
        }


def summarize_reduction(
    tok: AugmentedTokenizer,
    corpus: ParallelCorpus,
    lang: str,
    view: TokenizerView,
    pivot: str = "en",
) -> ReductionSummary:
    """Measure the effect of one pack on one language, held-out corpus.

    The caller is responsible for passing an *evaluation* slice disjoint from
    the mining slice.  Measuring reduction on the mining corpus would report the
    miner's memorisation, not a saving a user would see.

    Claim: reduction — benchmark metrics (1) and (5).
    """
    sents = corpus.by_lang[lang]
    pivot_sents = corpus.pivot_for(lang, pivot) if lang != pivot else None
    before = measure_fertility(tok, sents, lang, BASE_VIEW, pivot_sents)
    after = measure_fertility(tok, sents, lang, view, pivot_sents)
    n_new = len(view.allowed_ids - frozenset(range(tok.base_vocab_size))) if view.allowed_ids else 0
    return ReductionSummary(
        lang=lang,
        tokens_before=before.total_tokens,
        tokens_after=after.total_tokens,
        parity_ratio_before=before.parity_ratio,
        parity_ratio_after=after.parity_ratio,
        n_sentences=len(sents),
        n_new_tokens=n_new,
    )
