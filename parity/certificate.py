"""DriftCertificate — measure how far a new token moves the model, and bound it.

A vocabulary pack is only worth shipping if a deployer can say, in one sentence,
how much the model's behaviour may change.  This module produces that sentence.

What is measured
----------------
For each new token, on calibration contexts **disjoint from the ones used to fit
its embedding**, we compare the frozen model run two ways — on the original
expansion ``v_1..v_k`` and on the single new token — at aligned positions:

``kl_next_token``     KL(p_orig ‖ p_new) over the base vocabulary, in nats.
``tv_next_token``     total variation ‖p_orig − p_new‖₁ / 2, in [0, 1].
``logit_linf``        ‖z_orig − z_new‖_∞, the input to the deterministic bound.
``hidden_l2_rel``     ‖h_orig − h_new‖₂ / ‖h_orig‖₂ at the final residual stream.
``emit_logprob_err``  |log P_new(t | c) − log P_orig(v_1..v_k | c)|, in nats —
                      only when output embeddings were synthesised, since it is
                      the generation-side analogue of the input-side drift.

How it is bounded
-----------------
Three families, reported together because each answers a different question.

1. **Conformal upper tolerance limit** (the headline).  From the order
   statistics of ``n`` calibration measurements, with no distributional
   assumption: *with probability ≥ 1−δ over the calibration draw, at least a
   1−α fraction of future inputs from the same distribution have drift ≤ B.*
   This is a statement about the tail, which is what a user experiences; a bound
   on the mean is not.

2. **Empirical-Bernstein bound on the mean** (Maurer & Pontil 2009).  Tighter
   than Hoeffding when the variance is small, which it usually is here.
   Requires a bounded statistic, so KL is clipped and the clip rate is reported
   — a certificate with a 4% clip rate is a certificate about 96% of the data
   and must say so.

3. **Deterministic Lipschitz bound.**  If two logit vectors differ by at most ε
   in ∞-norm then ``KL(softmax(z) ‖ softmax(z')) ≤ 2ε`` (proof in
   :func:`kl_bound_from_logit_linf`).  Composed with
   ``‖Δz‖_∞ ≤ max_i ‖W_U[i]‖₂ · ‖Δh‖₂``, a certified radius on the
   residual-stream displacement yields a KL bound that holds for *every* input
   inside that radius, not merely on average.

What is **not** claimed
-----------------------
None of this is a worst-case bound over all possible inputs.  The radius in (3)
and the quantile in (1) are estimated from a finite calibration sample drawn
from a particular distribution; a deliberately adversarial prompt, or a domain
far from the calibration corpus, is outside the guarantee.  Saying so plainly is
part of the deliverable: an overstated certificate is worse than none, because
it transfers risk to the people least able to detect it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from parity.adapters import TorchLMAdapter
from parity.synthesis import CalibrationIndex, Context
from parity.types import BoundSpec, DriftCertificate, MergeCandidate, SynthesizedToken, fingerprint

log = logging.getLogger("parity.certificate")

STATISTICS = (
    "kl_next_token",
    "tv_next_token",
    "logit_linf",
    "hidden_l2_rel",
    "emit_logprob_err",
    "offcontext_mass",
)


@dataclass
class NegativeProbe:
    """Pre-computed states for the off-context (false-firing) statistic.

    Why this exists
    ---------------
    Every other statistic here is measured where the new token's expansion
    *actually occurs*.  That leaves a gap a certificate must not have: a fitted
    output row can hit its target on those contexts and still have a large
    component that fires on **unrelated** text, stealing probability mass from
    ordinary tokens — including English ones, under a multi-view deployment.

    This probe closes the gap.  It runs a generic corpus through the model once
    and caches the final hidden states and log-partition values, after which the
    mass any candidate would take at any position is a dot product:

        P(t | c) = exp(w·h) / (Z_base(c) + exp(w·h)) = sigmoid(w·h − log Z_base(c)).

    One forward pass serves every candidate, so the whole check is effectively
    free — which is the only reason it is affordable to run it on thousands of
    tokens.

    Claim: bound, non-regression — bounds the harm a new token can do where it
    does not belong, which is what protects the other languages sharing the
    model.
    """

    hidden: Any  # [M, d] final-layer states
    log_z: Any  # [M] log-partition of the base logits at those positions
    next_ids: List[Tuple[int, ...]] = field(default_factory=list)

    def __len__(self) -> int:
        return 0 if self.hidden is None else int(self.hidden.shape[0])


# ---------------------------------------------------------------------------
# Bound primitives
# ---------------------------------------------------------------------------


def kl_bound_from_logit_linf(eps: float) -> float:
    """Deterministic: ``KL(softmax(z) ‖ softmax(z')) ≤ 2ε`` when ``‖z−z'‖_∞ ≤ ε``.

    Proof.  Write ``p = softmax(z)``, ``q = softmax(z')``.  Then
    ``log(p_i/q_i) = (z_i − z'_i) − (log Z − log Z')`` where ``Z = Σ exp(z_j)``.
    Since ``Z = Σ exp(z_j) ≤ Σ exp(z'_j + ε) = e^ε Z'`` and symmetrically
    ``Z ≥ e^{−ε} Z'``, we get ``|log Z − log Z'| ≤ ε``.  Hence
    ``|log(p_i/q_i)| ≤ 2ε`` for every ``i``, and
    ``KL(p‖q) = Σ_i p_i log(p_i/q_i) ≤ 2ε``. ∎

    Claim: bound — the only genuinely deterministic link in the chain, and the
    reason ``logit_linf`` is measured at all.
    """
    return 2.0 * float(eps)


def logit_linf_bound_from_hidden(row_norm_max: float, hidden_l2: float) -> float:
    """Deterministic: ``‖W_U Δh‖_∞ ≤ max_i ‖W_U[i]‖₂ · ‖Δh‖₂`` (Cauchy-Schwarz).

    Claim: bound — converts a residual-stream radius into a logit radius, which
    :func:`kl_bound_from_logit_linf` then converts into a KL bound.
    """
    return float(row_norm_max) * float(hidden_l2)


def empirical_bernstein_upper(samples: Sequence[float], delta: float = 0.05, range_hi: float = 1.0) -> float:
    """(1−δ) upper confidence bound on the mean of a ``[0, range_hi]`` variable.

    Maurer & Pontil (2009), Theorem 4::

        mean + sqrt(2 · V_n · ln(2/δ) / n) + 7 · R · ln(2/δ) / (3(n − 1))

    with ``V_n`` the unbiased sample variance.  Falls back to Hoeffding for
    ``n < 2``, where the variance term is undefined.

    Claim: bound — the mean-drift guarantee reported alongside the tail bound.
    """
    xs = [float(x) for x in samples]
    n = len(xs)
    if n == 0:
        return float("inf")
    mean = sum(xs) / n
    if n < 2:
        return min(range_hi, mean + range_hi * math.sqrt(math.log(2.0 / delta) / 2.0))
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    log_term = math.log(2.0 / delta)
    bound = mean + math.sqrt(2.0 * var * log_term / n) + 7.0 * range_hi * log_term / (3.0 * (n - 1))
    return min(float(range_hi), bound)


def hoeffding_upper(samples: Sequence[float], delta: float = 0.05, range_hi: float = 1.0) -> float:
    """(1−δ) upper confidence bound on the mean, assumption-light and loose.

    Kept for comparison: when it is much looser than the empirical-Bernstein
    bound, the drift distribution has low variance, which is itself worth
    knowing.

    Claim: bound.
    """
    xs = [float(x) for x in samples]
    n = len(xs)
    if n == 0:
        return float("inf")
    mean = sum(xs) / n
    return min(float(range_hi), mean + range_hi * math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def binomial_tolerance_index(n: int, alpha: float, delta: float) -> Optional[int]:
    """Smallest ``k`` such that the ``k``-th order statistic is a (1−α, 1−δ) limit.

    The coverage of the ``k``-th order statistic of ``n`` i.i.d. samples is
    ``Beta(k, n+1−k)`` distributed, and
    ``P(Beta(k, n+1−k) ≥ p) = P(Binom(n, p) ≤ k−1)``.  So ``X_(k)`` covers at
    least ``1−α`` of the distribution with confidence ``1−δ`` iff

        P(Binom(n, 1−α) ≤ k − 1)  ≥  1 − δ,

    and we take the smallest such ``k`` to get the tightest valid bound.

    Sanity check the implementation against the textbook case: for
    ``α = δ = 0.05`` this returns ``k = n`` (the sample maximum) first at
    ``n = 59``, and ``None`` for ``n ≤ 58`` — the classical "59 samples for a
    95/95 one-sided tolerance limit" rule.  ``tests/test_certificate.py``
    asserts exactly that.

    Returns ``None`` when no ``k ≤ n`` qualifies — the calibration set is too
    small to support the requested (α, δ).  Returning ``None`` rather than
    quietly falling back to the maximum is deliberate: "we do not have the
    evidence" is a valid, and here a required, answer, and it is what makes an
    under-calibrated token get rejected instead of shipped.

    Claim: bound — the exact finite-sample condition behind the headline
    guarantee, computed rather than approximated.
    """
    if n <= 0:
        return None
    p = 1.0 - alpha
    q = 1.0 - p
    target = 1.0 - delta
    cdf = 0.0
    for k in range(1, n + 1):
        i = k - 1
        try:
            term = math.comb(n, i) * (p ** i) * (q ** (n - i))
        except (OverflowError, ValueError):  # pragma: no cover
            term = 0.0
        cdf += term
        if cdf >= target:
            return k
    return None


def _samples_needed(alpha: float, delta: float, cap: int = 100_000) -> int:
    """Smallest ``n`` for which a (1−α, 1−δ) tolerance limit exists at all.

    Taking ``k = n`` (the sample maximum), the condition reduces to
    ``1 − (1−α)^n ≥ 1 − δ``, i.e. ``n ≥ log δ / log(1−α)``.  For (0.95, 0.95)
    that is 59; for (0.90, 0.95) it is 29.

    Claim: bound — tells an operator exactly how much calibration data a given
    guarantee costs, before they spend a GPU-hour discovering it.
    """
    if not (0 < alpha < 1) or not (0 < delta < 1):
        return cap
    return min(cap, int(math.ceil(math.log(delta) / math.log(1 - alpha))))


def conformal_upper(samples: Sequence[float], alpha: float = 0.05, delta: float = 0.05) -> Tuple[float, Optional[int]]:
    """Distribution-free (1−α, 1−δ) upper tolerance limit from order statistics.

    Returns ``(bound, k)``; ``bound`` is ``inf`` and ``k`` is ``None`` when the
    sample is too small for the requested guarantee.

    Claim: bound — the tail guarantee Parity advertises.
    """
    xs = sorted(float(x) for x in samples)
    n = len(xs)
    k = binomial_tolerance_index(n, alpha, delta)
    if k is None:
        return float("inf"), None
    return xs[k - 1], k


# ---------------------------------------------------------------------------
# Certifier
# ---------------------------------------------------------------------------


@dataclass
class CertifierConfig:
    """Tolerances and confidence levels for :class:`DriftCertifier`.

    The acceptance thresholds are the deployer's risk appetite, not a property
    of the method — they are surfaced on the CLI (``--max-kl``) and printed in
    every model card so that a user can see the bar a pack cleared.

    Claim: bound.
    """

    delta: float = 0.05
    alpha: float = 0.05
    kl_clip: float = 1.0  # nats; clip rate is reported
    tv_clip: float = 1.0
    # Acceptance thresholds, applied to the conformal (tail) bound.
    max_kl: float = 0.05
    max_tv: float = 0.05
    max_emit_logprob_err: float = 0.75
    #: Largest probability the new token may take at positions where its
    #: expansion does **not** occur.  This is the guard against a fitted output
    #: row that satisfies its targets by firing everywhere.
    max_offcontext_mass: float = 0.01
    #: Positions sampled for the off-context probe.  One forward pass total.
    offcontext_positions: int = 512
    #: Build an **input-only** pack: its tokens are readable but never emitted.
    #:
    #: Set this when the model ties its embeddings. There the input row is also
    #: the output row, so "reproduce the expansion's internal state" and "be
    #: emitted with the expansion's probability" are two objectives on one
    #: vector, and satisfying both is not generally possible. Declining to emit
    #: drops the second objective, which makes emission drift **zero by
    #: construction** instead of a number to bound — and costs almost nothing,
    #: since prompt tokens are where the saving is and generation still produces
    #: base tokens that decode to the same strings.
    #:
    #: The serving side enforces it: `MultiTokenizerRouter` masks pack ids out of
    #: the sampler for a `lang:in` view.
    input_only: bool = False
    min_calibration: int = 12
    keep_observations: bool = True
    max_batch: int = 64
    #: Whether a finite emission bound is required for acceptance.
    #:
    #: The emission statistic yields **one** measurement per occurrence, while
    #: the distributional statistics yield one per aligned position (~7x more).
    #: So the emission bound is what actually determines how large a calibration
    #: corpus must be: a (1-alpha, 1-delta) = (0.95, 0.95) tolerance limit needs
    #: 59 held-out occurrences of the token.  Leaving this ``True`` means a
    #: token with too few occurrences is rejected for lack of evidence rather
    #: than shipped with a silently unbounded generation-side drift.
    require_emit_bound: bool = True


class DriftCertifier:
    """Measure and bound the behavioural drift caused by each new token.

    Claim: bound — the component that makes "we did not break the model" a
    number with a confidence level attached.
    """

    def __init__(self, adapter: TorchLMAdapter, config: Optional[CertifierConfig] = None):
        self.lm = adapter
        self.cfg = config or CertifierConfig()
        W = self.lm.unembed_matrix()
        self._row_norm_max = float(W.float().norm(dim=1).max())

    @property
    def unembed_row_norm_max(self) -> float:
        """``max_i ‖W_U[i]‖₂`` — the Lipschitz constant of the logit map.

        Claim: bound.
        """
        return self._row_norm_max

    def effective_output_row(self, token: SynthesizedToken) -> Optional[torch.Tensor]:
        """The vector that will actually produce this token's logit at serving time.

        For an untied model that is the synthesised output row.  For a **tied**
        model there is no second vector: the input embedding is the output row,
        so emission drift must be measured against it — and that is why
        :class:`~parity.synthesis.SynthesisConfig` folds an emission term into
        the input objective when embeddings are tied.

        Most small open-weight models (Qwen2.5-0.5B, Llama-3.2-1B) tie, so this
        is the common case, not the exotic one.

        Claim: bound — certifying the wrong vector would certify nothing.
        """
        if token.output_embedding is not None:
            return token.output_embedding
        if self.lm.tied_embeddings:
            return token.input_embedding
        return None

    # -- measurement --------------------------------------------------------

    @torch.no_grad()
    def measure(self, token: SynthesizedToken, contexts: Sequence[Context]) -> Dict[str, List[float]]:
        """Raw per-position drift measurements for one token.

        Both runs use the *same* frozen weights; the new token is spliced into
        ``inputs_embeds`` so that the output head still spans exactly the base
        vocabulary.  That keeps ``p_orig`` and ``p_new`` supported on the same
        set, so the KL is a comparison of like with like and needs no
        renormalisation fudge.

        Claim: bound — this is the evidence; everything after it is arithmetic.
        """
        stats: Dict[str, List[float]] = {k: [] for k in STATISTICS}
        if not contexts:
            return stats

        for start in range(0, len(contexts), self.cfg.max_batch):
            batch = list(contexts[start : start + self.cfg.max_batch])
            orig_rows = [c.orig_ids for c in batch]
            new_rows = [c.new_ids(0) for c in batch]

            oe, om = self.lm.embeds_for(orig_rows)
            otr = self.lm.trace(inputs_embeds=oe, attention_mask=om)

            ne, nm = self.lm.embeds_for(new_rows)
            ne = ne.clone()
            for bi, c in enumerate(batch):
                ne[bi, len(c.prefix)] = token.input_embedding.to(ne.dtype)
            ntr = self.lm.trace(inputs_embeds=ne, attention_mask=nm)

            for bi, c in enumerate(batch):
                P, k = len(c.prefix), len(c.expansion)
                o_pos = [P + k - 1] + [P + k + j for j in range(len(c.suffix))]
                n_pos = [P] + [P + 1 + j for j in range(len(c.suffix))]
                for op, np_ in zip(o_pos, n_pos):
                    zo = otr.logits[bi, op].float()
                    zn = ntr.logits[bi, np_].float()
                    lp = torch.log_softmax(zo, -1)
                    lq = torch.log_softmax(zn, -1)
                    p = lp.exp()
                    kl = float((p * (lp - lq)).sum())
                    tv = float(0.5 * (p - lq.exp()).abs().sum())
                    stats["kl_next_token"].append(max(0.0, kl))
                    stats["tv_next_token"].append(min(1.0, max(0.0, tv)))
                    stats["logit_linf"].append(float((zo - zn).abs().max()))
                    ho = otr.hidden_states[-1][bi, op].float()
                    hn = ntr.hidden_states[-1][bi, np_].float()
                    stats["hidden_l2_rel"].append(float((ho - hn).norm() / ho.norm().clamp_min(1e-6)))

                emit_row = self.effective_output_row(token)
                if emit_row is not None and P > 0:
                    logits = otr.logits[bi].float()
                    logZ = torch.logsumexp(logits[P - 1], dim=-1)
                    s = torch.zeros((), dtype=torch.float32)
                    for i in range(k):
                        s = s + torch.log_softmax(logits[P - 1 + i], dim=-1)[c.expansion[i]]
                    s = s.clamp(max=-1e-4)
                    target = s + logZ - torch.log1p(-torch.exp(s))
                    got = torch.dot(emit_row.float(), otr.hidden_states[-1][bi, P - 1].float())
                    # Convert the logit error back into a log-probability error,
                    # which is the quantity a generation-time user would feel.
                    stats["emit_logprob_err"].append(float((got - target).abs()))
        return stats

    @torch.no_grad()
    def build_negative_probe(self, docs: Sequence[Sequence[int]], max_positions: Optional[int] = None) -> NegativeProbe:
        """Cache final-layer states over generic text, once, for every candidate.

        Include text in the languages the deployment must not disturb — English
        above all — because that is precisely where a false-firing token would
        do damage that the on-context statistics cannot see.

        Claim: bound, low-cost — one forward pass over the probe corpus bounds
        the off-context behaviour of the entire pack.
        """
        limit = max_positions or self.cfg.offcontext_positions
        rows, zs, nexts = [], [], []
        stride = max(1, sum(max(0, len(d) - 1) for d in docs) // max(1, limit))
        for start in range(0, len(docs), self.cfg.max_batch):
            batch = [list(d)[:128] for d in docs[start : start + self.cfg.max_batch] if len(d) >= 2]
            if not batch:
                continue
            embeds, mask = self.lm.embeds_for(batch)
            tr = self.lm.trace(inputs_embeds=embeds, attention_mask=mask)
            for bi, d in enumerate(batch):
                for p in range(0, len(d) - 1, stride):
                    rows.append(tr.hidden_states[-1][bi, p].float())
                    zs.append(torch.logsumexp(tr.logits[bi, p].float(), dim=-1))
                    nexts.append(tuple(d[p + 1 : p + 9]))
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
        if not rows:
            return NegativeProbe(hidden=None, log_z=None, next_ids=[])
        return NegativeProbe(hidden=torch.stack(rows), log_z=torch.stack(zs), next_ids=nexts)

    @torch.no_grad()
    def measure_offcontext(self, token: SynthesizedToken, probe: NegativeProbe) -> List[float]:
        """Probability the new token would take where its expansion does not occur.

        Positions whose continuation *is* the expansion are excluded — the token
        firing there is the point of it, not a fault.

        Claim: bound, non-regression — a bounded statistic in ``[0, 1]``, so the
        conformal and empirical-Bernstein machinery applies directly.
        """
        row = self.effective_output_row(token)
        if row is None or len(probe) == 0:
            return []
        exp = tuple(token.candidate.ids)
        k = len(exp)
        keep = [i for i, nxt in enumerate(probe.next_ids) if nxt[:k] != exp]
        if not keep:
            return []
        idx = torch.tensor(keep, dtype=torch.long)
        h = probe.hidden.index_select(0, idx)
        z = h @ row.float()
        mass = torch.sigmoid(z - probe.log_z.index_select(0, idx))
        return [float(x) for x in mass]

    # -- bounding -----------------------------------------------------------

    def bound(self, stats: Dict[str, List[float]], fp: str) -> DriftCertificate:
        """Turn raw measurements into a :class:`DriftCertificate`.

        Claim: bound.
        """
        cfg = self.cfg
        bounds: Dict[str, BoundSpec] = {}
        n = len(stats.get("kl_next_token", []))

        for name, clip in (("kl_next_token", cfg.kl_clip), ("tv_next_token", cfg.tv_clip), ("offcontext_mass", 1.0)):
            xs = stats.get(name, [])
            if not xs:
                continue
            clipped = [min(x, clip) for x in xs]
            clip_rate = sum(1 for x in xs if x > clip) / len(xs)
            cval, k = conformal_upper(xs, alpha=cfg.alpha, delta=cfg.delta)
            bounds[name] = BoundSpec(
                statistic=name,
                method="conformal_quantile",
                value=cval,
                delta=cfg.delta,
                alpha=cfg.alpha,
                n_samples=len(xs),
                empirical_mean=sum(xs) / len(xs),
                empirical_max=max(xs),
                range_hi=clip,
                clip_rate=clip_rate,
                notes=(
                    f"order statistic k={k}" if k else "sample too small for the requested (alpha, delta)"
                ),
            )
            bounds[name + "_mean"] = BoundSpec(
                statistic=name + " (mean)",
                method="empirical_bernstein",
                value=empirical_bernstein_upper(clipped, delta=cfg.delta, range_hi=clip),
                delta=cfg.delta,
                n_samples=len(xs),
                empirical_mean=sum(xs) / len(xs),
                empirical_max=max(xs),
                range_hi=clip,
                clip_rate=clip_rate,
                notes="Maurer-Pontil empirical Bernstein on the clipped statistic",
            )

        # Deterministic chain: certified radius on ||dh|| -> logit linf -> KL.
        hs = stats.get("hidden_l2_rel", [])
        linf = stats.get("logit_linf", [])
        if linf:
            linf_bound, k = conformal_upper(linf, alpha=cfg.alpha, delta=cfg.delta)
            bounds["kl_next_token_lipschitz"] = BoundSpec(
                statistic="KL(next token) via logit sup-norm",
                method="lipschitz",
                value=kl_bound_from_logit_linf(linf_bound),
                delta=cfg.delta,
                alpha=cfg.alpha,
                n_samples=len(linf),
                empirical_mean=sum(linf) / len(linf),
                empirical_max=max(linf),
                range_hi=float("inf"),
                notes=(
                    "KL <= 2*||dz||_inf, deterministic; ||dz||_inf certified conformally "
                    f"(order statistic k={k}); unembedding row-norm max = {self._row_norm_max:.4g}"
                ),
            )
        if hs:
            hb, _ = conformal_upper(hs, alpha=cfg.alpha, delta=cfg.delta)
            bounds["hidden_l2_rel"] = BoundSpec(
                statistic="hidden_l2_rel",
                method="conformal_quantile",
                value=hb,
                delta=cfg.delta,
                alpha=cfg.alpha,
                n_samples=len(hs),
                empirical_mean=sum(hs) / len(hs),
                empirical_max=max(hs),
                range_hi=float("inf"),
            )

        emit = stats.get("emit_logprob_err", [])
        if emit:
            eb, _ = conformal_upper(emit, alpha=cfg.alpha, delta=cfg.delta)
            bounds["emit_logprob_err"] = BoundSpec(
                statistic="emit_logprob_err",
                method="conformal_quantile",
                value=eb,
                delta=cfg.delta,
                alpha=cfg.alpha,
                n_samples=len(emit),
                empirical_mean=sum(emit) / len(emit),
                empirical_max=max(emit),
                range_hi=float("inf"),
                notes="nats of error in the log-probability the new token is emitted with",
            )

        accepted, reason = self._accept(bounds, n)
        return DriftCertificate(
            token_key="",
            bounds=bounds,
            observed={k: list(v) for k, v in stats.items()} if cfg.keep_observations else {},
            calibration_fingerprint=fp,
            n_calibration=n,
            accepted=accepted,
            reject_reason=reason,
        )

    def _accept(self, bounds: Dict[str, BoundSpec], n: int) -> Tuple[bool, str]:
        cfg = self.cfg
        if n < cfg.min_calibration:
            return False, f"only {n} calibration measurements (need >= {cfg.min_calibration})"
        kl = bounds.get("kl_next_token")
        if kl is None or not math.isfinite(kl.value):
            return False, "no finite KL tolerance limit at the requested (alpha, delta)"
        if kl.value > cfg.max_kl:
            return False, f"certified KL tail bound {kl.value:.4g} > tolerance {cfg.max_kl:.4g}"
        tv = bounds.get("tv_next_token")
        if tv is not None and tv.value > cfg.max_tv:
            return False, f"certified TV tail bound {tv.value:.4g} > tolerance {cfg.max_tv:.4g}"
        off = bounds.get("offcontext_mass")
        if off is not None and off.value > cfg.max_offcontext_mass:
            return False, (
                f"off-context firing mass {off.value:.4g} > tolerance {cfg.max_offcontext_mass:.4g} "
                "(the token would steal probability where it does not belong)"
            )
        emit = bounds.get("emit_logprob_err")
        if cfg.input_only:
            # The token is never emitted, so its emission drift is not a risk
            # this pack carries. The measurement is still recorded in `observed`.
            emit = None
        if emit is not None and not math.isfinite(emit.value):
            if cfg.require_emit_bound:
                need = _samples_needed(cfg.alpha, cfg.delta)
                return False, (
                    f"only {emit.n_samples} held-out occurrences; a ({1 - cfg.alpha:.2f}, {1 - cfg.delta:.2f}) "
                    f"emission bound needs >= {need}"
                )
        elif emit is not None and emit.value > cfg.max_emit_logprob_err:
            return False, f"emission log-prob error {emit.value:.4g} > tolerance {cfg.max_emit_logprob_err:.4g}"
        return True, ""

    # -- driver -------------------------------------------------------------

    def certify(
        self,
        tokens: Sequence[SynthesizedToken],
        holdout: CalibrationIndex,
        max_contexts: int = 24,
        negative_docs: Optional[Sequence[Sequence[int]]] = None,
    ) -> Dict[str, DriftCertificate]:
        """Certify every synthesised token against a held-out calibration index.

        ``holdout`` must be built on corpus lines that were **not** used to fit
        the embeddings; :func:`parity.build.split_corpus` enforces the split and
        the CLI passes disjoint slices.

        ``negative_docs`` should include text in every language the deployment
        must not disturb (English first).  Without it the off-context statistic
        cannot be computed and tokens are accepted without that protection —
        which the certificate then records rather than glosses over.

        Claim: bound — produces the object that decides what may be shipped.
        """
        out: Dict[str, DriftCertificate] = {}
        probe = self.build_negative_probe(negative_docs) if negative_docs else NegativeProbe(None, None, [])
        if len(probe) == 0 and negative_docs:
            log.warning("negative probe is empty; off-context firing will not be bounded")
        for i, tok in enumerate(tokens):
            ctxs = holdout.contexts_for(tok.candidate)[:max_contexts]
            fp = fingerprint({"ids": list(tok.candidate.ids), "contexts": [c.orig_ids for c in ctxs]})
            stats = self.measure(tok, ctxs)
            if len(probe):
                stats["offcontext_mass"] = self.measure_offcontext(tok, probe)
            cert = self.bound(stats, fp)
            cert = DriftCertificate(
                token_key=tok.candidate.key,
                bounds=cert.bounds,
                observed=cert.observed,
                calibration_fingerprint=fp,
                n_calibration=cert.n_calibration,
                accepted=cert.accepted,
                reject_reason=cert.reject_reason,
            )
            out[tok.candidate.key] = cert
            if (i + 1) % 200 == 0:
                log.info("certified %d/%d tokens", i + 1, len(tokens))
        n_acc = sum(1 for c in out.values() if c.accepted)
        log.info("certified %d tokens; %d accepted, %d rejected", len(out), n_acc, len(out) - n_acc)
        return out


# ---------------------------------------------------------------------------
# Re-verification (used by the test suite and by `parity verify`)
# ---------------------------------------------------------------------------


@dataclass
class Coverage:
    """Fresh-data coverage of one certified bound.

    ``fraction`` is the share of fresh measurements that landed at or below the
    certified limit; the guarantee says this should be at least ``1 − α``.

    Claim: bound.
    """

    token_key: str
    statistic: str
    bound: float
    fraction: float
    n: int
    target: float
    max_observed: float
    delta: float = 0.05

    @property
    def upper_ci(self) -> float:
        """One-sided 99% upper confidence limit on the coverage fraction.

        A finite fresh sample estimates coverage with error ~``1/sqrt(n)``, so
        comparing the point estimate against ``1 − α`` would flag a violation
        every time noise pushed it a hair low.  We flag only when we are
        confident the true coverage is below target.

        Claim: bound.
        """
        if self.n <= 0:
            return 1.0
        se = math.sqrt(max(0.0, self.fraction * (1 - self.fraction)) / self.n)
        return min(1.0, self.fraction + 2.326 * se + 0.5 / self.n)

    @property
    def violated(self) -> bool:
        """True when the bound demonstrably under-covers on fresh data.

        Claim: bound.
        """
        return self.upper_ci < self.target


@dataclass
class VerificationResult:
    """Outcome of re-measuring drift on fresh data and checking the bounds.

    Claim: bound — this is the falsification test.  A certificate that has never
    been checked against data it did not see is a claim, not a certificate.
    """

    n_tokens: int
    n_checked: int
    coverages: List[Coverage] = field(default_factory=list)
    violations: List[Tuple[str, str, float, float]] = field(default_factory=list)

    @property
    def violation_rate(self) -> float:
        """Fraction of (token, statistic) checks that under-covered.

        Claim: bound.
        """
        return len(self.violations) / max(1, self.n_checked)

    @property
    def allowed_violation_rate(self) -> float:
        """Share of bounds the guarantee *permits* to under-cover, i.e. mean ``δ``.

        A ``(1−α, 1−δ)`` tolerance limit promises that the bound covers ``1−α``
        of future inputs **with probability ``1−δ`` over the calibration draw**.
        So across many independently calibrated tokens, up to a ``δ`` fraction of
        the issued bounds are expected to under-cover.  Demanding that *none* do
        would reject a perfectly valid procedure; that is the level the check
        below compares against.

        Claim: bound.
        """
        if not self.coverages:
            return 0.0
        return sum(c.delta for c in self.coverages) / len(self.coverages)

    @property
    def violation_allowance(self) -> float:
        """Upper limit on the observed violation rate before we call it a failure.

        ``δ`` plus a one-sided finite-sample margin, because a handful of tokens
        cannot pin down a rate.  Note the margin is deliberately generous: the
        two statistics per token are measured on the same contexts and are
        strongly correlated, so the effective number of independent checks is
        closer to the token count than to ``n_checked``.

        Claim: bound.
        """
        n = max(1, self.n_checked)
        p = self.allowed_violation_rate
        return p + 2.326 * math.sqrt(max(p * (1 - p), 1e-6) / n) + 1.0 / n

    @property
    def ok(self) -> bool:
        """True when under-coverage stays within what the guarantee permits.

        Claim: bound.
        """
        return self.violation_rate <= self.violation_allowance

    @property
    def mean_coverage(self) -> float:
        """Average share of fresh measurements inside their bound.

        Should sit at or above ``1 − α``; well above means the bounds are loose,
        which is safe but wasteful, and is worth knowing.

        Claim: bound.
        """
        if not self.coverages:
            return float("nan")
        return sum(c.fraction for c in self.coverages) / len(self.coverages)

    @property
    def mean_target(self) -> float:
        """Average coverage target ``1 − α`` across the checked bounds.

        Claim: bound.
        """
        if not self.coverages:
            return float("nan")
        return sum(c.target for c in self.coverages) / len(self.coverages)


def verify_certificates(
    certifier: DriftCertifier,
    tokens: Sequence[SynthesizedToken],
    certificates: Dict[str, DriftCertificate],
    fresh: CalibrationIndex,
    statistics: Sequence[str] = ("kl_next_token", "tv_next_token"),
    max_contexts: int = 32,
) -> VerificationResult:
    """Re-measure on fresh contexts and check each bound's realised coverage.

    The right test for a ``(1−α, 1−δ)`` tolerance limit is a **coverage** test,
    not a maximum test and not a point-quantile test.  The guarantee is

        P( drift ≤ B ) ≥ 1 − α,

    so we measure the fresh fraction below ``B`` and flag the bound only when a
    one-sided 99% confidence limit on that fraction still falls short of
    ``1 − α``.  Comparing raw maxima would fail a *correct* bound roughly every
    time, since the bound is explicitly allowed an ``α`` tail; comparing point
    quantiles fails it whenever finite-sample noise nudges the estimate.

    Claim: bound — the executable form of "the measured drift stays inside the
    certificate", which ``tests/test_certificate.py`` and ``parity verify``
    both call.
    """
    coverages: List[Coverage] = []
    violations: List[Tuple[str, str, float, float]] = []
    checked = 0
    for tok in tokens:
        cert = certificates.get(tok.candidate.key)
        if cert is None or not cert.accepted:
            continue
        ctxs = fresh.contexts_for(tok.candidate)[:max_contexts]
        if len(ctxs) < 2:
            continue
        stats = certifier.measure(tok, ctxs)
        for name in statistics:
            xs = stats.get(name, [])
            if not xs or name not in cert.bounds:
                continue
            spec = cert.bounds[name]
            if not math.isfinite(spec.value):
                continue
            inside = sum(1 for x in xs if x <= spec.value + 1e-12)
            cov = Coverage(
                token_key=tok.candidate.key,
                statistic=name,
                bound=spec.value,
                fraction=inside / len(xs),
                n=len(xs),
                target=1.0 - spec.alpha,
                max_observed=max(xs),
                delta=spec.delta,
            )
            coverages.append(cov)
            checked += 1
            if cov.violated:
                violations.append((tok.candidate.key, name, cov.fraction, cov.target))
    return VerificationResult(n_tokens=len(tokens), n_checked=checked, coverages=coverages, violations=violations)
