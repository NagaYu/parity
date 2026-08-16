"""VocabularySelection — spend a fixed embedding budget on the best tokens.

The problem
-----------
After mining and certification we have thousands of *admissible* candidates and
room for a few thousand embedding rows.  Ranking by ``count × (len − 1)`` is
wrong, because candidates overlap: adopting "の研究" and "研究の" does not save
the sum of their individual savings, since only one of them can match at a given
position under leftmost-longest tokenization.

The formulation
---------------
Lay the corpus out as a flat array of token slots.  A merge of length ``L`` at
position ``p`` removes exactly the ``L − 1`` *internal boundaries* it spans, so
define the ground set of elements as those boundaries and let

    F(S) = | ⋃_{t ∈ S} ⋃_{occurrences of t} boundaries(occurrence) |.

``F`` is a weighted coverage function: monotone, non-negative, and submodular.
Cardinality-constrained maximisation of such an ``F`` admits the classical
greedy ``(1 − 1/e)`` guarantee (Nemhauser, Wolsey & Fisher 1978), implemented
here with CELF lazy evaluation so that a 200k-candidate ground set is seconds,
not hours.

Surrogate vs. truth, stated honestly
------------------------------------
``F`` is an **upper bound** on the tokens actually saved: when two adopted
merges overlap, the union counts both sets of boundaries but the tokenizer can
only apply one.  So we report three numbers, never one:

``surrogate``       ``F(S)`` — what the optimiser maximised.
``exact_savings``   tokens actually removed, by retokenising the corpus with
                    the selected pack.  This is the number quoted anywhere else
                    in the project.
``online_bound``    ``F(S) + Σ top-B marginal gains at S`` — a *data-dependent*
                    upper bound on ``F(OPT)``, and therefore on the exact
                    savings of any admissible pack of the same size.  Dividing
                    ``exact_savings`` by it gives a certified optimality ratio
                    for the run, which is usually far better than the worst-case
                    ``1 − 1/e``.

Only certified candidates are eligible: :meth:`VocabularySelector.select` takes
the certificate map and drops anything whose drift bound exceeded tolerance.
The budget therefore constrains *reduction subject to a proven bound*, which is
the optimisation problem the project set out to solve.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from parity.types import DriftCertificate, MergeCandidate

log = logging.getLogger("parity.selection")


@dataclass
class SelectionConfig:
    """Budget and filters for :class:`VocabularySelector`.

    Claim: reduction, low-cost — ``budget`` is literally the number of embedding
    rows the operator is willing to pay for.
    """

    budget: int = 4000
    min_marginal_gain: int = 1
    lazy: bool = True
    #: Cap on how many candidates the online bound sums over; ``budget`` by
    #: definition, exposed for tests.
    online_bound_terms: Optional[int] = None
    log_every: int = 500


@dataclass
class SelectionResult:
    """Chosen tokens plus every number needed to judge the choice.

    Claim: reduction — the object that becomes a vocabulary pack.
    """

    selected: List[MergeCandidate] = field(default_factory=list)
    gains: List[int] = field(default_factory=list)
    surrogate: int = 0
    exact_savings: int = 0
    baseline_tokens: int = 0
    online_bound: float = 0.0
    n_ground_set: int = 0
    n_rejected_by_certificate: int = 0

    @property
    def token_reduction(self) -> float:
        """Exact fraction of corpus tokens removed by the selected pack.

        Claim: reduction — the x-axis of the Pareto figure, measured not
        estimated.
        """
        if self.baseline_tokens <= 0:
            return 0.0
        return self.exact_savings / self.baseline_tokens

    @property
    def certified_optimality_ratio(self) -> float:
        """``exact_savings / online_bound`` — a data-dependent optimality ratio.

        Because ``online_bound ≥ F(OPT) ≥ exact_savings(OPT)``, this is a valid
        lower bound on how close the greedy pack is to the best pack of the same
        size.  It is typically well above the worst-case ``1 − 1/e ≈ 0.632``.

        Claim: reduction — turns "greedy is fine" into a checkable number for
        this run, on this corpus, at this budget.
        """
        if self.online_bound <= 0:
            return 0.0
        return min(1.0, self.exact_savings / self.online_bound)

    @property
    def savings_per_row(self) -> float:
        """Tokens saved per embedding row spent.

        Claim: reduction, low-cost — the efficiency of the vocabulary budget.
        """
        if not self.selected:
            return 0.0
        return self.exact_savings / len(self.selected)

    def to_dict(self) -> Dict[str, object]:
        """Serialise for the run manifest and the model card.

        Claim: infrastructure.
        """
        return {
            "n_selected": len(self.selected),
            "n_ground_set": self.n_ground_set,
            "n_rejected_by_certificate": self.n_rejected_by_certificate,
            "surrogate": self.surrogate,
            "exact_savings": self.exact_savings,
            "baseline_tokens": self.baseline_tokens,
            "token_reduction": self.token_reduction,
            "online_bound": self.online_bound,
            "certified_optimality_ratio": self.certified_optimality_ratio,
            "greedy_worst_case_guarantee": 1.0 - 1.0 / np.e,
            "savings_per_row": self.savings_per_row,
        }


# ---------------------------------------------------------------------------
# Occurrence indexing
# ---------------------------------------------------------------------------


def boundary_index(
    docs: Sequence[Sequence[int]], candidates: Sequence[MergeCandidate]
) -> Tuple[List[np.ndarray], int]:
    """Map each candidate to the flat boundary slots its occurrences cover.

    One trie walk over the corpus records *every* match at every position
    (nested and overlapping included), so the cost is ``O(corpus · max_len)``
    for the whole ground set rather than ``O(|candidates| · corpus)``.

    Claim: reduction, low-cost — the data structure the submodular objective is
    evaluated on, built in a single pass.
    """
    from parity.tokenization import MergeTrie

    trie = MergeTrie()
    for i, c in enumerate(candidates):
        trie.add(c.ids, i)

    offsets: List[int] = []
    total = 0
    for d in docs:
        offsets.append(total)
        total += len(d)

    per_cand: List[List[int]] = [[] for _ in candidates]
    for di, doc in enumerate(docs):
        base = offsets[di]
        n = len(doc)
        for i in range(n):
            node = trie
            j = i
            while j < n:
                child = node.children.get(doc[j])
                if child is None:
                    break
                node = child
                j += 1
                if node.token is not None:
                    # boundaries removed by matching [i, j) are i .. j-2
                    per_cand[node.token].extend(range(base + i, base + j - 1))
    arrays = [np.unique(np.asarray(v, dtype=np.int64)) if v else np.zeros(0, dtype=np.int64) for v in per_cand]
    return arrays, total


def exact_savings(docs: Sequence[Sequence[int]], selected: Sequence[MergeCandidate]) -> int:
    """Tokens actually removed when the corpus is retokenised with ``selected``.

    Runs the real leftmost-longest merge pass, so overlap, nesting and
    tie-breaking are handled exactly as they will be at serving time.  This is
    the ground truth the surrogate is checked against.

    Claim: reduction — no reported reduction number in this repository comes
    from the surrogate; they all come from here.
    """
    from parity.tokenization import MergeTrie

    if not selected:
        return 0
    trie = MergeTrie()
    for i, c in enumerate(selected):
        trie.add(c.ids, i)
    saved = 0
    for doc in docs:
        saved += len(doc) - len(trie.merge(doc))
    return saved


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class VocabularySelector:
    """Greedy maximum-coverage selection under a budget and a drift filter.

    Claim: reduction, bound — maximises measured token savings over exactly the
    candidates whose behavioural drift has been certified within tolerance.
    """

    def __init__(
        self,
        candidates: Sequence[MergeCandidate],
        docs: Sequence[Sequence[int]],
        certificates: Optional[Dict[str, DriftCertificate]] = None,
        config: Optional[SelectionConfig] = None,
    ):
        self.cfg = config or SelectionConfig()
        self.docs = [list(d) for d in docs]
        certificates = certificates or {}
        self.n_rejected = 0
        eligible: List[MergeCandidate] = []
        for c in candidates:
            cert = certificates.get(c.key)
            if certificates and (cert is None or not cert.accepted):
                self.n_rejected += 1
                continue
            eligible.append(c)
        self.candidates = eligible
        self.cover, self.n_slots = boundary_index(self.docs, self.candidates)
        self.baseline_tokens = sum(len(d) for d in self.docs)
        log.info(
            "selection ground set: %d eligible (%d rejected by certificate), %d corpus tokens",
            len(self.candidates),
            self.n_rejected,
            self.baseline_tokens,
        )

    def select(self, budget: Optional[int] = None) -> SelectionResult:
        """Run CELF lazy greedy to the budget and report all three numbers.

        Claim: reduction — the component that decides what a pack contains.
        """
        budget = int(budget if budget is not None else self.cfg.budget)
        covered = np.zeros(self.n_slots, dtype=bool)
        chosen: List[int] = []
        gains: List[int] = []

        # Lazy greedy: the heap stores (-stale_gain, last_updated_round, index).
        heap: List[Tuple[int, int, int]] = []
        for i, arr in enumerate(self.cover):
            if arr.size:
                heapq.heappush(heap, (-int(arr.size), -1, i))
        round_id = 0

        while heap and len(chosen) < budget:
            neg_gain, stamp, idx = heapq.heappop(heap)
            arr = self.cover[idx]
            gain = int(np.count_nonzero(~covered[arr])) if arr.size else 0
            if gain < self.cfg.min_marginal_gain:
                continue
            if self.cfg.lazy and stamp != round_id and heap and gain < -heap[0][0]:
                # Stale: someone else may now be better. Re-insert with the
                # freshly computed (and therefore exact) gain.
                heapq.heappush(heap, (-gain, round_id, idx))
                continue
            covered[arr] = True
            chosen.append(idx)
            gains.append(gain)
            round_id += 1
            if len(chosen) % self.cfg.log_every == 0:
                log.info("selected %d/%d, surrogate=%d", len(chosen), budget, int(covered.sum()))

        selected = [self.candidates[i] for i in chosen]
        surrogate = int(covered.sum())
        online = self._online_bound(covered, set(chosen), budget)
        exact = exact_savings(self.docs, selected)
        result = SelectionResult(
            selected=selected,
            gains=gains,
            surrogate=surrogate,
            exact_savings=exact,
            baseline_tokens=self.baseline_tokens,
            online_bound=online,
            n_ground_set=len(self.candidates),
            n_rejected_by_certificate=self.n_rejected,
        )
        log.info(
            "selected %d tokens: exact saving %d/%d (%.1f%%), certified optimality >= %.3f",
            len(selected),
            exact,
            self.baseline_tokens,
            100 * result.token_reduction,
            result.certified_optimality_ratio,
        )
        return result

    def _online_bound(self, covered: np.ndarray, chosen: set, budget: int) -> float:
        """``F(S) + Σ top-B marginal gains at S`` — an upper bound on ``F(OPT)``.

        Valid for any monotone submodular ``F``: for any set ``T`` with
        ``|T| ≤ B``, submodularity gives
        ``F(T) ≤ F(S) + Σ_{t ∈ T} [F(S ∪ {t}) − F(S)]``, and the right-hand side
        is maximised by taking the ``B`` largest marginal gains.  Since the true
        token savings of any pack are at most its surrogate value, this bounds
        the achievable savings too.

        Claim: reduction — a certificate of near-optimality that is specific to
        this corpus and budget, rather than the worst-case constant.
        """
        k = self.cfg.online_bound_terms or budget
        remaining = []
        for i, arr in enumerate(self.cover):
            if i in chosen or not arr.size:
                continue
            g = int(np.count_nonzero(~covered[arr]))
            if g > 0:
                remaining.append(g)
        remaining.sort(reverse=True)
        return float(covered.sum() + sum(remaining[:k]))

    # -- diagnostics --------------------------------------------------------

    def budget_curve(self, budgets: Sequence[int]) -> List[Dict[str, float]]:
        """Exact reduction at several budgets, from a single greedy run.

        Greedy is nested — the first ``b`` picks of a run to budget ``B`` are
        exactly the run to budget ``b`` — so the whole curve costs one pass.
        This is what the "reduction vs. embedding budget" panel of the figure
        is built from.

        Claim: reduction, low-cost.
        """
        top = max(budgets)
        full = self.select(top)
        out = []
        for b in sorted(budgets):
            sel = full.selected[:b]
            ex = exact_savings(self.docs, sel)
            out.append(
                {
                    "budget": float(b),
                    "n_selected": float(len(sel)),
                    "exact_savings": float(ex),
                    "token_reduction": ex / max(1, self.baseline_tokens),
                }
            )
        return out


def naive_topk(candidates: Sequence[MergeCandidate], budget: int) -> List[MergeCandidate]:
    """Frequency×length ranking — the baseline the submodular objective beats.

    Included so the benchmark can quantify what the overlap-aware objective is
    worth, rather than assuming it is worth something.

    Claim: reduction — an ablation, and an honest one: on corpora with little
    candidate overlap the two agree, and the benchmark says so when they do.
    """
    return sorted(candidates, key=lambda c: (-c.raw_saving, c.key))[:budget]
