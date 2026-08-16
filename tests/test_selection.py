"""Selection: submodular greedy, and the three numbers it must report honestly.

The objective the optimiser maximises (boundary coverage) is an *upper bound* on
the tokens actually saved.  These tests pin down that relationship — surrogate ≥
exact, online bound ≥ surrogate — so that a future change cannot quietly start
reporting the surrogate as if it were the saving.
"""

from __future__ import annotations

import numpy as np
import pytest

from parity.selection import SelectionConfig, VocabularySelector, boundary_index, exact_savings, naive_topk
from parity.types import DriftCertificate, MergeCandidate


def _cand(ids, count=10, lang="xx"):
    return MergeCandidate(ids=tuple(ids), surface="".join(chr(96 + i) for i in ids), count=count, doc_count=count, lang=lang)


def test_exact_savings_accounts_for_overlap():
    docs = [[1, 2, 3, 4]]
    # Two overlapping merges: leftmost-longest can only apply one of them.
    a = _cand([1, 2, 3])
    b = _cand([2, 3, 4])
    assert exact_savings(docs, [a]) == 2
    assert exact_savings(docs, [b]) == 2
    assert exact_savings(docs, [a, b]) == 2, "overlapping merges must not double-count"


def test_boundary_coverage_upper_bounds_the_true_saving():
    docs = [[1, 2, 3, 4], [1, 2, 3, 4]]
    cands = [_cand([1, 2, 3]), _cand([2, 3, 4])]
    cover, n_slots = boundary_index(docs, cands)
    union = len(np.union1d(cover[0], cover[1]))
    assert union >= exact_savings(docs, cands)


def test_greedy_respects_the_budget_and_reports_consistent_numbers():
    rng = np.random.default_rng(0)
    docs = [list(rng.integers(1, 30, size=40)) for _ in range(60)]
    cands = []
    for _ in range(60):
        n = int(rng.integers(2, 5))
        cands.append(_cand(list(rng.integers(1, 30, size=n))))
    cands = list({c.ids: c for c in cands}.values())

    sel = VocabularySelector(cands, docs, None, SelectionConfig(budget=10))
    res = sel.select()
    assert len(res.selected) <= 10
    assert res.exact_savings == exact_savings(docs, res.selected)
    assert res.surrogate >= res.exact_savings
    assert res.online_bound >= res.surrogate
    assert 0.0 <= res.certified_optimality_ratio <= 1.0
    assert res.token_reduction == pytest.approx(res.exact_savings / res.baseline_tokens)


def test_greedy_marginal_gains_are_non_increasing():
    """Diminishing returns — the observable signature of submodularity."""
    rng = np.random.default_rng(1)
    docs = [list(rng.integers(1, 20, size=50)) for _ in range(40)]
    cands = list({c.ids: c for c in (_cand(list(rng.integers(1, 20, size=3))) for _ in range(50))}.values())
    res = VocabularySelector(cands, docs, None, SelectionConfig(budget=20)).select()
    gains = res.gains
    assert gains == sorted(gains, reverse=True), "lazy greedy must select in non-increasing marginal gain order"


def test_online_bound_dominates_any_alternative_selection():
    """The data-dependent certificate must actually bound competitors."""
    rng = np.random.default_rng(2)
    docs = [list(rng.integers(1, 15, size=40)) for _ in range(40)]
    cands = list({c.ids: c for c in (_cand(list(rng.integers(1, 15, size=3))) for _ in range(40))}.values())
    sel = VocabularySelector(cands, docs, None, SelectionConfig(budget=8))
    res = sel.select()
    for trial in range(20):
        alt = list(rng.permutation(len(cands))[:8])
        alt_saving = exact_savings(docs, [cands[i] for i in alt])
        assert alt_saving <= res.online_bound + 1e-9


def test_uncertified_candidates_are_excluded():
    docs = [[1, 2, 3, 4, 1, 2, 3, 4]]
    good, bad = _cand([1, 2]), _cand([3, 4])
    certs = {
        good.key: DriftCertificate(token_key=good.key, accepted=True),
        bad.key: DriftCertificate(token_key=bad.key, accepted=False, reject_reason="drift"),
    }
    sel = VocabularySelector([good, bad], docs, certs, SelectionConfig(budget=5))
    res = sel.select()
    assert [c.key for c in res.selected] == [good.key]
    assert res.n_rejected_by_certificate == 1


def test_budget_curve_is_monotone(built):
    from parity.selection import VocabularySelector as VS

    result = built["result"]
    docs = [built["tokenizer"].encode(l) for l in built["corpus"].lines[:80]]
    cands = [e.candidate for e in result.pack.entries]
    if len(cands) < 4:
        pytest.skip("fixture pack too small for a budget curve")
    sel = VS(cands, docs, None, SelectionConfig(budget=len(cands)))
    curve = sel.budget_curve([1, 2, max(3, len(cands) // 2), len(cands)])
    reductions = [c["token_reduction"] for c in curve]
    assert reductions == sorted(reductions), "more budget must never save fewer tokens"


def test_submodular_greedy_beats_or_matches_frequency_ranking():
    rng = np.random.default_rng(3)
    docs = [list(rng.integers(1, 12, size=60)) for _ in range(50)]
    cands = list({c.ids: c for c in (_cand(list(rng.integers(1, 12, size=3))) for _ in range(60))}.values())
    for c in cands:
        pass
    sel = VocabularySelector(cands, docs, None, SelectionConfig(budget=10))
    greedy = sel.select()
    naive = naive_topk(cands, 10)
    assert greedy.exact_savings >= exact_savings(docs, naive)
