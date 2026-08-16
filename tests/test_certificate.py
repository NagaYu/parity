"""The bound claim: measured drift stays inside the certificate.

This is the test the project's central promise rests on, so it is written to be
*falsifiable*: it re-measures drift on contexts the certificate never saw and
compares against the stored limits.

Two things it deliberately does not do:

* It does not assert that drift is *small*.  The fixture model is untrained; the
  claim under test is "the issued bound holds", not "the bound is impressive".
* It does not treat a single exceedance as a failure.  A (1−α) tolerance limit
  is *defined* to permit an α fraction of future samples above it, so the check
  is on the per-token empirical quantile, which is what the bound covers.

The fixture corpus reuses vocabulary across splits, so the distribution shift
between calibration and verification is milder than in real deployment.  That
makes this test necessary but not sufficient; ``parity verify`` exists so the
same check can be run on real corpora and real models.
"""

from __future__ import annotations

import math

import pytest

from parity.certificate import (
    DriftCertifier,
    _samples_needed,
    binomial_tolerance_index,
    conformal_upper,
    empirical_bernstein_upper,
    hoeffding_upper,
    kl_bound_from_logit_linf,
    verify_certificates,
)
from parity.corpora import expand_for_testing
from parity.synthesis import CalibrationIndex


# ---------------------------------------------------------------------------
# The bound primitives, against closed-form references
# ---------------------------------------------------------------------------


def test_tolerance_index_matches_the_textbook_95_95_rule():
    # The classical result: the maximum of 59 samples is a one-sided 95/95
    # distribution-free tolerance limit, and 58 samples are not enough.
    assert binomial_tolerance_index(58, 0.05, 0.05) is None
    assert binomial_tolerance_index(59, 0.05, 0.05) == 59
    # 90/95 needs 29.
    assert binomial_tolerance_index(28, 0.10, 0.05) is None
    assert binomial_tolerance_index(29, 0.10, 0.05) == 29
    # With more samples the required order statistic is no longer the maximum.
    k = binomial_tolerance_index(200, 0.05, 0.05)
    assert k is not None and k < 200


def test_samples_needed_agrees_with_the_tolerance_index():
    for alpha, delta in [(0.05, 0.05), (0.10, 0.05), (0.15, 0.10), (0.01, 0.05)]:
        n = _samples_needed(alpha, delta)
        assert binomial_tolerance_index(n, alpha, delta) is not None
        assert binomial_tolerance_index(n - 1, alpha, delta) is None


def test_conformal_upper_is_an_order_statistic_and_refuses_small_samples():
    xs = [i / 100 for i in range(100)]
    value, k = conformal_upper(xs, alpha=0.05, delta=0.05)
    assert k is not None and value in xs
    assert value >= sorted(xs)[int(0.9 * len(xs))]
    inf_value, none_k = conformal_upper(xs[:10], alpha=0.05, delta=0.05)
    assert none_k is None and math.isinf(inf_value)


def test_conformal_bound_covers_the_stated_fraction_empirically():
    # Draw calibration/test pairs from a heavy-tailed distribution and check the
    # coverage guarantee holds at the advertised rate.
    import random

    rng = random.Random(0)
    alpha, delta, n = 0.10, 0.05, 60
    failures = 0
    trials = 200
    for _ in range(trials):
        calib = [rng.paretovariate(1.5) for _ in range(n)]
        bound, _ = conformal_upper(calib, alpha=alpha, delta=delta)
        test = [rng.paretovariate(1.5) for _ in range(200)]
        covered = sum(1 for t in test if t <= bound) / len(test)
        if covered < 1 - alpha:
            failures += 1
    # The guarantee is: with prob >= 1-delta the coverage is >= 1-alpha.
    # Allow generous slack for the finite number of trials.
    assert failures / trials <= delta + 0.08, f"coverage failed on {failures}/{trials} draws"


def test_empirical_bernstein_is_valid_and_tighter_than_hoeffding_at_low_variance():
    xs = [0.10] * 50 + [0.11] * 50
    eb = empirical_bernstein_upper(xs, delta=0.05, range_hi=1.0)
    hf = hoeffding_upper(xs, delta=0.05, range_hi=1.0)
    assert eb >= sum(xs) / len(xs)
    assert eb < hf, "empirical Bernstein should win when the variance is tiny"


def test_kl_lipschitz_lemma_holds_numerically():
    import torch

    torch.manual_seed(0)
    for _ in range(200):
        z = torch.randn(64) * 3
        eps = float(torch.rand(1)) * 0.5
        z2 = z + (torch.rand(64) * 2 - 1) * eps
        lp, lq = torch.log_softmax(z, -1), torch.log_softmax(z2, -1)
        kl = float((lp.exp() * (lp - lq)).sum())
        assert kl <= kl_bound_from_logit_linf(float((z - z2).abs().max())) + 1e-6


# ---------------------------------------------------------------------------
# The end-to-end claim
# ---------------------------------------------------------------------------


def test_every_shipped_token_carries_an_accepted_finite_certificate(built):
    pack = built["result"].pack
    assert len(pack) > 0, "the fixture build produced no tokens; the test would be vacuous"
    for entry in pack.entries:
        cert = entry.certificate
        assert cert.accepted and not cert.reject_reason
        for name in ("kl_next_token", "tv_next_token"):
            assert name in cert.bounds
            assert math.isfinite(cert.bounds[name].value)
            assert cert.bounds[name].n_samples >= cert.bounds[name].alpha and cert.bounds[name].n_samples > 0
        assert cert.calibration_fingerprint, "a bound must be traceable to the data it came from"


def test_rejected_candidates_state_a_reason(built):
    certs = built["result"].certificates
    rejected = [c for c in certs.values() if not c.accepted]
    assert rejected, "the fixture tolerance should be binding on something"
    assert all(c.reject_reason for c in rejected)


def test_measured_drift_stays_within_the_certified_bound_on_fresh_contexts(built, sample, base_tokenizer):
    """The falsification test: re-measure on data the certificate never saw."""
    result = built["result"]
    adapter = built["adapter"]

    fresh_lines = expand_for_testing(sample.by_lang["ja"], 200, seed=1234)
    fresh_docs = [base_tokenizer.encode(l) for l in fresh_lines]
    shipped_keys = {e.candidate.key for e in result.pack.entries}
    tokens = [t for t in result.synthesized if t.candidate.key in shipped_keys]
    assert tokens

    index = CalibrationIndex([t.candidate for t in tokens], 12, 6)
    index.scan(fresh_docs, max_per_candidate=32)

    certifier = DriftCertifier(adapter, None)
    certs = {e.candidate.key: e.certificate for e in result.pack.entries}
    verdict = verify_certificates(certifier, tokens, certs, index)

    assert verdict.n_checked > 0, "nothing was re-measured; the test would be vacuous"
    assert verdict.ok, (
        f"under-coverage {100 * verdict.violation_rate:.1f}% exceeded the "
        f"{100 * verdict.violation_allowance:.1f}% the (1-alpha, 1-delta) guarantee permits: "
        + "; ".join(f"{k}/{s}: coverage {got:.3f} < target {tgt:.3f}" for k, s, got, tgt in verdict.violations[:5])
    )
    # And the bounds must not be vacuous in the other direction either: mean
    # coverage well below target would mean the limits are systematically thin.
    assert verdict.mean_coverage >= verdict.mean_target - 0.05


def test_certificate_is_falsifiable(built):
    """A bound that nothing can violate is not a bound: check `holds` bites."""
    cert = built["result"].pack.entries[0].certificate
    limit = cert.value("kl_next_token")
    assert cert.holds("kl_next_token", limit * 0.5)
    assert cert.holds("kl_next_token", limit)
    assert not cert.holds("kl_next_token", limit + 1.0)
    with pytest.raises(KeyError):
        cert.value("a_statistic_we_never_measured")


def test_calibration_splits_are_disjoint(corpus):
    from parity.build import split_corpus

    s = split_corpus(corpus)
    sets = [set(s.mine.lines), set(s.fit.lines), set(s.certify.lines), set(s.eval.lines)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "a bound fitted on the data it is checked against is not a bound"
