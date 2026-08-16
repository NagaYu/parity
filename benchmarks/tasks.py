"""Downstream metrics that are comparable *across different tokenizations*.

The measurement problem
-----------------------
Comparing a model before and after a vocabulary change is a trap: perplexity per
*token* falls automatically when tokens get longer, so a vocabulary pack can
"improve" a model by doing nothing useful.  Every metric here is therefore
defined on the underlying **string**, not on tokens:

``bits_per_character``
    ``−log₂ P(text) / |characters|``.  Tokenizer-invariant by construction, and
    the standard way to compare language models with different vocabularies.
    A pack that damages the model raises it; a pack that is behaviour-preserving
    leaves it flat while the token count falls.

``translation_retrieval_accuracy``
    Pick the correct target-language sentence for an English sentence out of
    ``1 + n_distractors`` candidates, scoring each by length-normalised
    conditional log-probability.  A real discriminative task, computable from
    the same parallel corpus with any causal LM and no extra downloads.

``english_regression``
    The same two metrics restricted to English, which for Parity must come out
    exactly zero and for continued pretraining generally does not.

One caveat stated rather than buried: an augmented vocabulary can express a
string with more than one token sequence, so scoring only the canonical
(leftmost-longest) sequence under-counts ``P(text)`` slightly, making
``bits_per_character`` a conservative *upper* bound for the pack conditions.
Conservative in the direction that works against Parity, which is the right
direction for a number we are using to advertise it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from parity.corpora import ParallelCorpus, count_chars, normalize

log = logging.getLogger("benchmarks.tasks")

LN2 = math.log(2.0)


@torch.no_grad()
def sequence_logprob(router, ids: Sequence[int], view: str, bos_id: int = 0) -> float:
    """Total log-probability of an id sequence under a view, in nats.

    A fixed one-token prefix is prepended so that the *first* content token is
    scored too; otherwise different tokenizations would leave different amounts
    of the string unscored and the comparison would be meaningless.

    Claim: non-regression — the primitive behind every quality metric here.
    """
    ids = list(ids)
    if not ids:
        return 0.0
    inp = torch.tensor([[bos_id] + ids], dtype=torch.long, device=router.lm.device)
    logits, _ = router.lm.run(inp, use_cache=False)
    mask = router.logit_mask(view).to(logits.device)
    lp = torch.log_softmax(logits[0, :-1].float() + mask, dim=-1)
    target = torch.tensor(ids, dtype=torch.long, device=logits.device)
    return float(lp.gather(1, target[:, None]).sum())


def bits_per_character(router, sentences: Sequence[str], view: str, bos_id: int = 0) -> float:
    """Bits per character of ``sentences`` under ``view`` — tokenizer-invariant.

    Lower is better.  This is the metric on which "the pack did not damage the
    model" is decided, and it cannot be gamed by making tokens longer.

    Claim: non-regression — benchmark metric (2).
    """
    total_nats = 0.0
    total_chars = 0
    for s in sentences:
        s = normalize(s)
        ids = router.encode(s, view)
        total_nats += -sequence_logprob(router, ids, view, bos_id)
        total_chars += count_chars(s)
    return total_nats / LN2 / max(1, total_chars)


def tokens_per_sentence(router, sentences: Sequence[str], view: str) -> float:
    """Mean token count — the cost side of the trade-off.

    Claim: reduction — benchmark metric (1).
    """
    if not sentences:
        return 0.0
    return sum(router.count(s, view) for s in sentences) / len(sentences)


@dataclass
class RetrievalResult:
    """Outcome of the translation-retrieval probe.

    Claim: non-regression — benchmark metric (2), the discriminative half.
    """

    accuracy: float
    n_items: int
    n_choices: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialise.

        Claim: infrastructure.
        """
        return {"accuracy": self.accuracy, "n_items": self.n_items, "n_choices": self.n_choices}


@torch.no_grad()
def translation_retrieval_accuracy(
    router,
    corpus: ParallelCorpus,
    lang: str,
    view: str,
    pivot: str = "en",
    n_distractors: int = 7,
    template: str = "{src}\n{tgt}",
    seed: int = 0,
    max_items: Optional[int] = None,
) -> RetrievalResult:
    """Can the model still tell which target sentence matches the English one?

    Each candidate is scored by its length-normalised conditional log-probability
    given the English sentence in a fixed template.  Length normalisation is by
    *characters*, not tokens, so the metric does not reward the pack for shorter
    sequences — the point is to measure comprehension, not compression.

    Claim: non-regression — a task that degrades if the synthesised embeddings
    are wrong, and is insensitive to token count if they are right.
    """
    import random

    rng = random.Random(seed)
    srcs = corpus.pivot_for(lang, pivot)
    tgts = corpus.by_lang[lang]
    n = min(len(srcs), len(tgts))
    if max_items:
        n = min(n, max_items)
    if n < 2:
        return RetrievalResult(accuracy=float("nan"), n_items=0, n_choices=0)
    k = min(n_distractors, n - 1)

    correct = 0
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        rng.shuffle(pool)
        choices = [i] + pool[:k]
        best, best_idx = -float("inf"), -1
        for c in choices:
            text = template.format(src=srcs[i], tgt=tgts[c])
            prefix_len = len(router.encode(template.format(src=srcs[i], tgt=""), view))
            ids = router.encode(text, view)
            total = sequence_logprob(router, ids, view)
            prefix = sequence_logprob(router, ids[:prefix_len], view) if prefix_len else 0.0
            score = (total - prefix) / max(1, count_chars(tgts[c]))
            if score > best:
                best, best_idx = score, c
        correct += int(best_idx == i)
    return RetrievalResult(accuracy=correct / n, n_items=n, n_choices=k + 1)


@dataclass
class QualityReport:
    """All quality numbers for one (condition, language) cell.

    Claim: reduction, non-regression — the two axes of the Pareto figure come
    straight out of this object.
    """

    condition: str
    lang: str
    view: str
    tokens_per_sentence: float
    token_reduction: float
    bits_per_character: float
    bpc_delta: float
    retrieval_accuracy: float
    retrieval_delta: float
    english_bpc: float
    english_bpc_delta: float
    english_tokens: float
    english_bpc_via_pack_view: float = float("nan")
    n_new_tokens: int = 0
    certified_kl_bound: Optional[float] = None
    build_flops: float = 0.0
    build_seconds: float = 0.0
    provenance: str = "measured"

    @property
    def quality_retention(self) -> float:
        """Share of baseline quality retained, in ``[0, 1]``.

        Defined from the retrieval task (an accuracy, so a ratio is meaningful)
        and clipped at 1: a condition that scores *above* baseline is reported as
        1.0 rather than >1, since the claim being made is preservation, not
        improvement.

        Claim: non-regression — the y-axis of the Pareto figure.
        """
        base = self.retrieval_accuracy - self.retrieval_delta
        if base <= 0:
            return float("nan")
        return min(1.0, self.retrieval_accuracy / base)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise.

        Claim: infrastructure.
        """
        d = dict(self.__dict__)
        d["quality_retention"] = self.quality_retention
        return d


def evaluate_condition(
    router,
    corpus: ParallelCorpus,
    lang: str,
    view: str,
    condition: str,
    baseline: Optional[QualityReport] = None,
    max_items: Optional[int] = 48,
    pivot: str = "en",
) -> QualityReport:
    """Run every metric for one condition and diff against the baseline.

    Claim: reduction, non-regression — produces one row of the results table,
    with both claims measured on the same forward passes.
    """
    tgt = corpus.by_lang[lang]
    eng = corpus.pivot_for(lang, pivot)
    tps = tokens_per_sentence(router, tgt, view)
    bpc = bits_per_character(router, tgt, view)
    retr = translation_retrieval_accuracy(router, corpus, lang, view, pivot=pivot, max_items=max_items)
    # English is scored under the **base** view, because that is how a router
    # serves it: the view is a per-request property, so an English request never
    # touches a Japanese pack. For condition (B) this still moves, since
    # continued pretraining rewrote the shared weights — which is exactly the
    # difference the benchmark exists to show.
    eng_bpc = bits_per_character(router, eng, "base")
    eng_tok = tokens_per_sentence(router, eng, "base")
    # Reported separately: what an operator would pay if they routed English
    # *through* the pack view anyway. Non-zero for Parity, and bounded by the
    # off-context certificate.
    eng_via_pack = bits_per_character(router, eng, view) if view != "base" else eng_bpc

    return QualityReport(
        condition=condition,
        lang=lang,
        view=view,
        tokens_per_sentence=tps,
        token_reduction=(0.0 if baseline is None else 1.0 - tps / max(1e-9, baseline.tokens_per_sentence)),
        bits_per_character=bpc,
        bpc_delta=(0.0 if baseline is None else bpc - baseline.bits_per_character),
        retrieval_accuracy=retr.accuracy,
        retrieval_delta=(0.0 if baseline is None else retr.accuracy - baseline.retrieval_accuracy),
        english_bpc=eng_bpc,
        english_bpc_delta=(0.0 if baseline is None else eng_bpc - baseline.english_bpc),
        english_tokens=eng_tok,
        english_bpc_via_pack_view=eng_via_pack,
    )
