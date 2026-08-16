"""FLOP and wall-clock accounting for benchmark metric (4).

The headline cost claim is a *ratio*, so it is only as good as its denominator.
Three rules keep it honest:

1. **Charge Parity the unfavourable rate.**  Forward passes are ``2N`` FLOPs per
   token, backward passes ``6N``; Parity's stages are forward-only and are
   charged at ``2N``, except when the optional Adam refinement runs, where the
   full ``6N`` is charged.

2. **Give the baseline a real recipe.**  Continued pretraining is measured at
   whatever scale the benchmark actually ran, and *separately* extrapolated to
   the 1–10B-token budgets published recipes use.  The extrapolation is labelled
   ``provenance="extrapolated"`` everywhere it appears, including on the figure.

3. **Count everything Parity spends.**  Mining, shortlisting, synthesis,
   certification and selection are itemised, not just the interesting one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Token budgets used by published vocabulary-adaptation recipes, for the
#: extrapolated comparison.  Cited as a range because practice varies widely.
REFERENCE_CPT_TOKENS = (1_000_000_000, 10_000_000_000)


def forward_flops(n_params: int, tokens: int) -> float:
    """``2 · N · tokens`` — the standard forward-pass estimate.

    Claim: low-cost.
    """
    return 2.0 * n_params * tokens


def training_flops(n_params: int, tokens: int) -> float:
    """``6 · N · tokens`` — forward plus backward, the standard training estimate.

    Claim: low-cost.
    """
    return 6.0 * n_params * tokens


@dataclass
class CostComparison:
    """Parity's build cost against continued pretraining, itemised.

    Claim: low-cost — benchmark metric (4), in a form a reader can audit line by
    line rather than take on trust.
    """

    parity_flops: float
    parity_seconds: float
    parity_stages: List[Dict[str, Any]] = field(default_factory=list)
    cpt_measured_flops: float = 0.0
    cpt_measured_tokens: int = 0
    cpt_measured_seconds: float = 0.0
    cpt_reference: List[Dict[str, Any]] = field(default_factory=list)
    n_params: int = 0
    added_params: int = 0

    @property
    def ratio_vs_measured(self) -> float:
        """How many times cheaper Parity was than the continued-pretraining run
        that was actually executed here.

        Claim: low-cost.
        """
        if self.parity_flops <= 0:
            return float("inf")
        return self.cpt_measured_flops / self.parity_flops

    def ratio_vs_reference(self, tokens: int) -> float:
        """Cost ratio against a realistic continued-pretraining budget.

        Extrapolated on the baseline's side only; Parity's number stays measured.

        Claim: low-cost.
        """
        if self.parity_flops <= 0:
            return float("inf")
        return training_flops(self.n_params, tokens) / self.parity_flops

    @property
    def orders_of_magnitude(self) -> float:
        """``log10`` of the cost ratio at the low end of the reference range.

        Reported at the *low* end so the advertised figure is the conservative
        one.

        Claim: low-cost.
        """
        import math

        r = self.ratio_vs_reference(REFERENCE_CPT_TOKENS[0])
        return math.log10(r) if r > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise, with provenance attached to every extrapolated figure.

        Claim: infrastructure.
        """
        return {
            "n_params": self.n_params,
            "added_params": self.added_params,
            "added_param_fraction": self.added_params / max(1, self.n_params),
            "parity": {
                "flops": self.parity_flops,
                "seconds": self.parity_seconds,
                "stages": self.parity_stages,
                "provenance": "measured",
            },
            "continued_pretraining_measured": {
                "flops": self.cpt_measured_flops,
                "tokens": self.cpt_measured_tokens,
                "seconds": self.cpt_measured_seconds,
                "provenance": "measured",
            },
            "continued_pretraining_reference": self.cpt_reference,
            "ratio_vs_measured": self.ratio_vs_measured,
            "ratio_vs_reference_1b": self.ratio_vs_reference(REFERENCE_CPT_TOKENS[0]),
            "ratio_vs_reference_10b": self.ratio_vs_reference(REFERENCE_CPT_TOKENS[1]),
            "orders_of_magnitude_conservative": self.orders_of_magnitude,
        }


def compare_costs(build_result, training_result=None, n_params: int = 0, added_params: int = 0) -> CostComparison:
    """Assemble the cost comparison from a build and (optionally) a training run.

    Claim: low-cost — benchmark metric (4).
    """
    n_params = n_params or 0
    ref = [
        {
            "tokens": t,
            "flops": training_flops(n_params, t),
            "provenance": "extrapolated",
            "basis": f"6*N*tokens with N={n_params}",
        }
        for t in REFERENCE_CPT_TOKENS
    ]
    return CostComparison(
        parity_flops=build_result.total_flops,
        parity_seconds=build_result.total_seconds,
        parity_stages=[c.to_dict() for c in build_result.costs],
        cpt_measured_flops=(training_result.flops if training_result else 0.0),
        cpt_measured_tokens=(training_result.tokens_seen if training_result else 0),
        cpt_measured_seconds=(training_result.seconds if training_result else 0.0),
        cpt_reference=ref,
        n_params=n_params,
        added_params=added_params,
    )
