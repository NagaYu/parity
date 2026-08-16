"""MergeCandidateMiner — find the token sequences a language is over-paying for.

Given a monolingual corpus and a base tokenizer, mine contiguous base-token
n-grams that occur often enough to be worth a vocabulary slot.

Algorithm
---------
Apriori-style level-wise counting with pruning.  An n-gram can only be frequent
if its (n-1)-prefix *and* its (n-1)-suffix are frequent, so we count level 2,
prune, extend only surviving prefixes to level 3, and so on.  This keeps memory
proportional to the number of *surviving* candidates rather than to the number
of possible n-grams, which matters as soon as the corpus is larger than a toy.

Filters (each exists because of a specific failure it prevents)
---------------------------------------------------------------
``min_count`` / ``min_doc_count``
    A sequence that appears 400 times in one boilerplate line is not a word.

``round_trip``
    ``base_encode(base_decode(ids)) == ids`` must hold, otherwise the token's
    surface form is not something the base tokenizer would ever produce, and the
    synthesised embedding would be fitted against contexts that never occur.

``max_length``
    Longer merges save more per occurrence but generalise worse and drift more;
    the certificate stage would reject them anyway, so we avoid paying to
    synthesise them.

``forbid_crossing``
    Optionally reject candidates whose surface spans a sentence boundary or a
    newline — they memorise corpus formatting rather than language.

What this module does *not* do
------------------------------
It does not decide what to adopt.  Ranking by ``count * (len - 1)`` double-counts
overlapping candidates; the overlap-aware objective lives in
:mod:`parity.selection`.  The miner's job is to produce a generous, clean
ground set.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from parity.corpora import Corpus
from parity.tokenization import BaseTokenizer
from parity.types import MergeCandidate

log = logging.getLogger("parity.miner")

_CROSSING = re.compile(r"[\n\r\t]")


@dataclass
class MinerConfig:
    """Knobs for :class:`MergeCandidateMiner`.

    Defaults are tuned for a FLORES-sized corpus (~1k sentences); the CLI raises
    ``min_count`` automatically for larger corpora via :meth:`auto_min_count`.

    Claim: reduction — every field trades ground-set size against candidate
    quality, and the comments say which way.
    """

    min_length: int = 2
    max_length: int = 8
    min_count: int = 3
    min_doc_count: int = 2
    max_candidates: int = 200_000
    require_round_trip: bool = True
    forbid_crossing: bool = True
    #: Drop candidates whose surface is pure whitespace/punctuation: they save
    #: tokens but make the tokenization brittle for no linguistic gain.
    min_alpha_chars: int = 1

    def auto_min_count(self, n_docs: int) -> int:
        """Scale ``min_count`` with corpus size so the ground set stays sane.

        Claim: low-cost — bounds the number of candidates we must certify, which
        is the dominant cost of a Parity build.
        """
        return max(self.min_count, int(round(n_docs / 400)))


class MergeCandidateMiner:
    """Extract and score merge candidates from a monolingual corpus.

    Claim: reduction — produces the ground set whose adoption is what actually
    lowers a language's token bill.
    """

    def __init__(self, tokenizer: BaseTokenizer, config: Optional[MinerConfig] = None):
        self.tok = tokenizer
        self.cfg = config or MinerConfig()

    # -- public API ---------------------------------------------------------

    def mine(self, corpus: Corpus, lang: Optional[str] = None) -> List[MergeCandidate]:
        """Mine candidates from ``corpus``, returned in descending raw saving.

        Claim: reduction — the returned list is the complete set of places this
        language is currently spending tokens it need not spend.
        """
        lang = lang or corpus.lang
        docs = [self.tok.encode(line) for line in corpus.lines]
        docs = [d for d in docs if len(d) >= self.cfg.min_length]
        if not docs:
            return []
        min_count = self.cfg.auto_min_count(len(docs))
        log.info("mining %s: %d docs, %d base tokens, min_count=%d", lang, len(docs), sum(map(len, docs)), min_count)

        counts, doc_counts = self._level_wise_count(docs, min_count)
        cands = self._materialize(counts, doc_counts, lang)
        cands.sort(key=lambda c: (-c.raw_saving, -c.count, c.key))
        if len(cands) > self.cfg.max_candidates:
            cands = cands[: self.cfg.max_candidates]
        log.info("mined %d candidates for %s (top saving %d tokens)", len(cands), lang, cands[0].raw_saving if cands else 0)
        return cands

    def baseline_token_count(self, corpus: Corpus) -> int:
        """Total base tokens in ``corpus`` — the denominator for reduction.

        Claim: reduction.
        """
        return sum(len(self.tok.encode(line)) for line in corpus.lines)

    # -- internals ----------------------------------------------------------

    def _level_wise_count(
        self, docs: Sequence[Sequence[int]], min_count: int
    ) -> Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], int]]:
        """Apriori level-wise n-gram counting with prefix/suffix pruning.

        Claim: low-cost — the pruning is what keeps mining a seconds-scale step
        rather than a memory-bound one, which matters for the cost claim (4).
        """
        counts: Dict[Tuple[int, ...], int] = {}
        doc_counts: Dict[Tuple[int, ...], int] = {}

        # Level 1: unigram frequency, needed to prune level 2.
        uni = Counter()
        for d in docs:
            uni.update(d)
        frequent: Set[Tuple[int, ...]] = {(t,) for t, c in uni.items() if c >= min_count}

        for n in range(2, self.cfg.max_length + 1):
            level = Counter()
            level_docs: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
            for di, d in enumerate(docs):
                if len(d) < n:
                    continue
                for i in range(len(d) - n + 1):
                    gram = tuple(d[i : i + n])
                    # Apriori pruning: both (n-1)-subgrams must be frequent.
                    if gram[:-1] not in frequent or gram[1:] not in frequent:
                        continue
                    level[gram] += 1
                    level_docs[gram].add(di)
            survivors = {
                g: c
                for g, c in level.items()
                if c >= min_count and len(level_docs[g]) >= self.cfg.min_doc_count
            }
            if not survivors:
                log.debug("level %d empty; stopping", n)
                break
            counts.update(survivors)
            doc_counts.update({g: len(level_docs[g]) for g in survivors})
            frequent = set(survivors)
            log.debug("level %d: %d survivors", n, len(survivors))
        return counts, doc_counts

    def _materialize(
        self,
        counts: Dict[Tuple[int, ...], int],
        doc_counts: Dict[Tuple[int, ...], int],
        lang: str,
    ) -> List[MergeCandidate]:
        """Apply surface-level filters and build :class:`MergeCandidate` s.

        Claim: reduction — the filters here are what separate "an id n-gram" from
        "a token a speaker would recognise as a unit".
        """
        out: List[MergeCandidate] = []
        n_rt, n_cross, n_alpha = 0, 0, 0
        for gram, count in counts.items():
            surface = self.tok.decode(gram)
            if not surface:
                continue
            if self.cfg.forbid_crossing and _CROSSING.search(surface):
                n_cross += 1
                continue
            if self.cfg.min_alpha_chars and sum(1 for ch in surface if ch.isalnum()) < self.cfg.min_alpha_chars:
                n_alpha += 1
                continue
            if self.cfg.require_round_trip and tuple(self.tok.encode(surface)) != gram:
                # The base tokenizer would not produce this id sequence for this
                # string, so contexts built from `surface` would not exercise the
                # expansion we are certifying against.
                n_rt += 1
                continue
            out.append(
                MergeCandidate(
                    ids=gram,
                    surface=surface,
                    count=count,
                    doc_count=doc_counts.get(gram, 0),
                    lang=lang,
                )
            )
        if n_rt or n_cross or n_alpha:
            log.info(
                "filtered candidates: %d non-round-tripping, %d crossing, %d non-alphanumeric", n_rt, n_cross, n_alpha
            )
        return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class MiningReport:
    """Where a language's tokens are going, before anything is adopted.

    Claim: reduction — quantifies the headroom, which is what makes the later
    "we recovered X% of it" statement interpretable.
    """

    lang: str
    n_docs: int
    n_base_tokens: int
    n_candidates: int
    top: List[MergeCandidate]
    upper_bound_saving: int

    @property
    def headroom(self) -> float:
        """Loose upper bound on the achievable token reduction, in ``[0, 1)``.

        Loose because it sums per-candidate savings and ignores overlap; the
        real, overlap-aware number comes out of :mod:`parity.selection`.  It is
        reported as an upper bound and labelled as such wherever it is printed.

        Claim: reduction.
        """
        if self.n_base_tokens <= 0:
            return 0.0
        return min(1.0, self.upper_bound_saving / self.n_base_tokens)

    def to_dict(self) -> Dict[str, object]:
        """Serialise for the run manifest.

        Claim: infrastructure.
        """
        return {
            "lang": self.lang,
            "n_docs": self.n_docs,
            "n_base_tokens": self.n_base_tokens,
            "n_candidates": self.n_candidates,
            "upper_bound_saving": self.upper_bound_saving,
            "headroom_upper_bound": self.headroom,
            "top": [c.to_dict() for c in self.top],
        }


def mining_report(
    miner: MergeCandidateMiner, corpus: Corpus, candidates: Sequence[MergeCandidate], top_k: int = 20
) -> MiningReport:
    """Summarise a mining run.

    Claim: reduction.
    """
    n_base = miner.baseline_token_count(corpus)
    return MiningReport(
        lang=corpus.lang,
        n_docs=len(corpus),
        n_base_tokens=n_base,
        n_candidates=len(candidates),
        top=list(candidates[:top_k]),
        upper_bound_saving=sum(c.raw_saving for c in candidates),
    )


def dedupe_nested(candidates: Iterable[MergeCandidate]) -> List[MergeCandidate]:
    """Drop a candidate when a strictly longer one covers all its occurrences.

    If ``ids`` occurs 40 times and every one of those occurrences is inside the
    same longer candidate, the short one can never be selected by leftmost-
    longest matching once the long one is adopted, so keeping it in the ground
    set only wastes certification budget.

    Claim: low-cost — shrinks the set that must be synthesised and certified,
    which is the dominant term in the build cost.
    """
    by_key = {c.ids: c for c in candidates}
    keep: List[MergeCandidate] = []
    for ids, cand in by_key.items():
        dominated = False
        for other_ids, other in by_key.items():
            if len(other_ids) <= len(ids):
                continue
            if _contains(other_ids, ids) and other.count >= cand.count:
                dominated = True
                break
        if not dominated:
            keep.append(cand)
    return keep


def _contains(haystack: Tuple[int, ...], needle: Tuple[int, ...]) -> bool:
    n, m = len(haystack), len(needle)
    return any(haystack[i : i + m] == needle for i in range(n - m + 1))
