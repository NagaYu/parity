"""Core data structures shared by every Parity component.

These are deliberately plain dataclasses with explicit ``to_dict``/``from_dict``
so that a vocabulary pack is a readable JSON document plus one safetensors
file — auditable by a language community that did not write this code.

Claim coverage: infrastructure (this module carries no experimental claim of
its own; it is the vocabulary in which the other claims are stated).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeCandidate:
    """One proposed new vocabulary entry: a contiguous run of base token ids.

    Parity works in *token-id space*, not string space.  A candidate is an
    n-gram over the base tokenizer's output; adopting it means "whenever this
    exact id sequence appears, emit one id instead of ``n``".  That choice
    makes encoding/decoding round-trip exactly for any base tokenizer (BPE,
    Unigram, WordPiece, byte-level) without touching pre-tokenizer internals.

    Attributes
    ----------
    ids:
        The base-token id sequence this candidate replaces (length >= 2).
    surface:
        ``base_tokenizer.decode(ids)``, kept for human review and for export to
        the Hugging Face ``added_tokens`` format.
    count:
        Number of occurrences observed in the mining corpus.
    doc_count:
        Number of distinct documents/lines the candidate occurred in; used to
        reject candidates that are frequent only inside one boilerplate line.
    raw_saving:
        ``count * (len(ids) - 1)`` — tokens removed if every occurrence merged
        and no occurrence overlapped another candidate.  An upper bound; the
        selection stage replaces it with a coverage objective.
    lang:
        ISO code of the corpus the candidate was mined from.

    Claim: reduction — a candidate is a concrete, measurable quantity of tokens
    that a target-language user currently pays for and need not.
    """

    ids: Tuple[int, ...]
    surface: str
    count: int
    doc_count: int = 0
    lang: str = ""

    def __post_init__(self) -> None:
        if len(self.ids) < 2:
            raise ValueError(f"a merge candidate must span >= 2 base tokens, got {self.ids!r}")

    @property
    def length(self) -> int:
        """Number of base tokens this candidate collapses.

        Claim: reduction — the per-occurrence saving is ``length - 1``.
        """
        return len(self.ids)

    @property
    def raw_saving(self) -> int:
        """Upper bound on tokens saved by adopting this candidate alone.

        Claim: reduction — used only to rank candidates before the (correct,
        overlap-aware) coverage objective in :mod:`parity.selection`.
        """
        return self.count * (self.length - 1)

    @property
    def key(self) -> str:
        """Stable identifier used as a dictionary key and in pack manifests.

        Claim: infrastructure.
        """
        return "-".join(str(i) for i in self.ids)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Claim: infrastructure.
        """
        return {
            "ids": list(self.ids),
            "surface": self.surface,
            "count": self.count,
            "doc_count": self.doc_count,
            "lang": self.lang,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergeCandidate":
        """Inverse of :meth:`to_dict`.

        Claim: infrastructure.
        """
        return cls(
            ids=tuple(int(i) for i in d["ids"]),
            surface=d.get("surface", ""),
            count=int(d.get("count", 0)),
            doc_count=int(d.get("doc_count", 0)),
            lang=d.get("lang", ""),
        )


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundSpec:
    """A single distribution-free guarantee about one measured statistic.

    Three bound families are used, and the ``method`` field says which:

    ``"empirical_bernstein"``
        High-probability upper bound on the **mean** of a bounded statistic
        (Maurer & Pontil 2009).  Reads: *with probability at least 1-delta over
        the draw of the calibration set, the true mean drift is <= value*.

    ``"conformal_quantile"``
        Distribution-free upper **tolerance** bound built from order statistics
        (the standard split-conformal argument).  Reads: *with probability at
        least 1-delta, at least a 1-alpha fraction of future inputs drawn from
        the same distribution have drift <= value*.  This is the bound Parity
        advertises, because a mean says little about the worst message a user
        will actually send.

    ``"lipschitz"``
        A **deterministic** bound: given a measured sup-norm on the residual
        stream perturbation and the row norms of the unembedding matrix, the
        KL between old and new next-token distributions is bounded by
        ``2 * ||W_U||_{2,inf} * ||delta_h||_2`` (see
        :func:`parity.certificate.kl_bound_from_logit_linf`).  It holds for
        *every* input whose hidden-state perturbation is within the measured
        radius — the radius itself is still empirical, which is why the
        conformal bound is reported alongside it.

    Claim: bound — this dataclass is the literal statement being certified.
    """

    statistic: str
    method: str
    value: float
    delta: float = 0.05
    alpha: float = 0.05
    n_samples: int = 0
    empirical_mean: float = 0.0
    empirical_max: float = 0.0
    range_hi: float = 1.0
    clip_rate: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Claim: infrastructure.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundSpec":
        """Inverse of :meth:`to_dict`.

        Claim: infrastructure.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def statement(self) -> str:
        """Render the guarantee as one English sentence for the model card.

        Claim: bound — a certificate nobody can read is not a certificate.
        """
        if self.method == "conformal_quantile":
            return (
                f"With probability >= {1 - self.delta:.2f} over the calibration draw, at least "
                f"{100 * (1 - self.alpha):.0f}% of future inputs from the same distribution have "
                f"{self.statistic} <= {self.value:.4g} (n={self.n_samples})."
            )
        if self.method == "empirical_bernstein":
            return (
                f"With probability >= {1 - self.delta:.2f}, the mean {self.statistic} is "
                f"<= {self.value:.4g} (n={self.n_samples}, empirical mean {self.empirical_mean:.4g})."
            )
        if self.method == "lipschitz":
            return (
                f"Deterministically, {self.statistic} <= {self.value:.4g} for every input whose "
                f"residual-stream perturbation stays within the measured radius "
                f"(n={self.n_samples} calibration probes)."
            )
        return f"{self.statistic} <= {self.value:.4g} ({self.method}, n={self.n_samples})."


@dataclass(frozen=True)
class DriftCertificate:
    """Everything measured about how far one new token moves the model.

    A certificate is *rejected* by :mod:`parity.selection` if any advertised
    bound exceeds the caller's tolerance.  Certificates are stored verbatim in
    the vocabulary pack and reprinted in the model card, so a downstream user
    can check the number they are being asked to accept.

    Attributes
    ----------
    token_key:
        :attr:`MergeCandidate.key` of the token this certificate describes.
    bounds:
        Mapping ``statistic -> BoundSpec``.  Always contains at least
        ``"kl_next_token"`` and ``"tv_next_token"``.
    observed:
        Raw per-sample measurements, kept so that a re-check can be run without
        re-deriving the calibration set.
    calibration_fingerprint:
        SHA-256 over the calibration inputs, so "the bound was computed on this
        data" is verifiable rather than asserted.
    accepted:
        Whether this token passed the tolerance used at build time.

    Claim: bound — this object *is* the guarantee.
    """

    token_key: str
    bounds: Dict[str, BoundSpec] = field(default_factory=dict)
    observed: Dict[str, List[float]] = field(default_factory=dict)
    calibration_fingerprint: str = ""
    n_calibration: int = 0
    accepted: bool = True
    reject_reason: str = ""

    def value(self, statistic: str) -> float:
        """Return the certified upper bound for ``statistic``.

        Claim: bound.
        """
        if statistic not in self.bounds:
            raise KeyError(f"no certified bound for {statistic!r}; have {sorted(self.bounds)}")
        return self.bounds[statistic].value

    def holds(self, statistic: str, measured: float, tol: float = 1e-9) -> bool:
        """Check a freshly measured value against the stored upper bound.

        This is the function the test suite calls: a certificate that cannot be
        falsified by new data is worthless, so we make falsifying it one call.

        Claim: bound — used by ``tests/test_certificate.py`` to verify that
        held-out drift really does land inside the advertised envelope.
        """
        return measured <= self.value(statistic) + tol

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (raw observations included).

        Claim: infrastructure.
        """
        return {
            "token_key": self.token_key,
            "bounds": {k: v.to_dict() for k, v in self.bounds.items()},
            "observed": {k: [float(x) for x in v] for k, v in self.observed.items()},
            "calibration_fingerprint": self.calibration_fingerprint,
            "n_calibration": self.n_calibration,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftCertificate":
        """Inverse of :meth:`to_dict`.

        Claim: infrastructure.
        """
        return cls(
            token_key=d["token_key"],
            bounds={k: BoundSpec.from_dict(v) for k, v in d.get("bounds", {}).items()},
            observed={k: list(v) for k, v in d.get("observed", {}).items()},
            calibration_fingerprint=d.get("calibration_fingerprint", ""),
            n_calibration=int(d.get("n_calibration", 0)),
            accepted=bool(d.get("accepted", True)),
            reject_reason=d.get("reject_reason", ""),
        )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@dataclass
class SynthesizedToken:
    """A new token with its synthesised embedding rows and fit diagnostics.

    ``input_embedding`` replaces the *k*-token expansion on the way in;
    ``output_embedding`` (``None`` when the model ties embeddings, or when the
    token is input-only) lets the model emit the new token.

    ``residual_before`` / ``residual_after`` are the least-squares objective
    values at initialisation (pure sub-token composition) and after the
    subspace Gauss-Newton solve.  Their ratio is the headline evidence that the
    optimisation step is doing real work.

    Claim: non-regression, low-cost — the embedding is obtained by solving a
    ``q``-dimensional least-squares problem on a handful of calibration
    contexts, not by training the model.
    """

    candidate: MergeCandidate
    input_embedding: Any  # torch.Tensor [d_model]
    output_embedding: Optional[Any] = None  # torch.Tensor [d_model] or None
    solver: str = "composition"
    residual_before: float = float("nan")
    residual_after: float = float("nan")
    n_calibration: int = 0
    seconds: float = 0.0
    flops: float = 0.0

    @property
    def residual_reduction(self) -> float:
        """Fraction of the composition-baseline residual removed by the solve.

        Claim: non-regression — larger is better; ~0 means the composition
        baseline was already at a local optimum for this token.
        """
        if not (self.residual_before > 0) or self.residual_after != self.residual_after:
            return 0.0
        return max(0.0, 1.0 - self.residual_after / self.residual_before)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metadata only; tensors live in the safetensors sidecar.

        Claim: infrastructure.
        """
        return {
            "candidate": self.candidate.to_dict(),
            "solver": self.solver,
            "residual_before": float(self.residual_before),
            "residual_after": float(self.residual_after),
            "n_calibration": self.n_calibration,
            "seconds": float(self.seconds),
            "flops": float(self.flops),
            "has_output_embedding": self.output_embedding is not None,
        }


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------


@dataclass
class FertilityReport:
    """Token cost of one (tokenizer, language) pair on a parallel corpus.

    Why three numbers instead of one:

    ``tokens_per_char``
        Works for every script, including ones without spaces.

    ``tokens_per_word``
        The classic "fertility"; only meaningful for whitespace-delimited
        writing.  ``None`` for Japanese and Thai unless a segmenter is supplied.

    ``parity_ratio``
        ``tokens(target sentence) / tokens(aligned English sentence)`` on a
        parallel corpus.  This is the number a user actually feels: how many
        times more of their context window, and of their bill, the same
        *meaning* costs.  It is well-defined for every script, which is why
        Parity leads with it.

    Claim: reduction — this is the baseline that the whole project is measured
    against, and the quantity the Space shows to a visitor.
    """

    lang: str
    tokenizer_id: str
    n_sentences: int
    total_tokens: int
    total_chars: int
    total_words: Optional[int] = None
    parity_ratio: Optional[float] = None
    parity_ratio_median: Optional[float] = None
    per_sentence_tokens: List[int] = field(default_factory=list)

    @property
    def tokens_per_char(self) -> float:
        """Tokens per NFC character — script-agnostic fertility.

        Claim: reduction.
        """
        return self.total_tokens / max(1, self.total_chars)

    @property
    def tokens_per_word(self) -> Optional[float]:
        """Tokens per whitespace word, or ``None`` for unsegmented scripts.

        Claim: reduction.
        """
        if not self.total_words:
            return None
        return self.total_tokens / self.total_words

    @property
    def effective_context_fraction(self) -> Optional[float]:
        """Share of the advertised context window a speaker actually receives.

        A ``parity_ratio`` of 2.5 means a 128k window holds roughly 51k
        English-equivalent tokens of this language: ``1 / 2.5``.

        Claim: reduction — converts an abstract ratio into the thing users lose.
        """
        if not self.parity_ratio:
            return None
        return 1.0 / self.parity_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Serialise, including the derived properties, for the atlas Dataset.

        Claim: infrastructure.
        """
        return {
            "lang": self.lang,
            "tokenizer_id": self.tokenizer_id,
            "n_sentences": self.n_sentences,
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "tokens_per_char": self.tokens_per_char,
            "tokens_per_word": self.tokens_per_word,
            "parity_ratio": self.parity_ratio,
            "parity_ratio_median": self.parity_ratio_median,
            "effective_context_fraction": self.effective_context_fraction,
        }


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------


@dataclass
class PackEntry:
    """One adopted token inside a :class:`VocabPack`.

    Claim: infrastructure — binds a candidate, its synthesised rows and its
    certificate into a single auditable unit.
    """

    candidate: MergeCandidate
    new_id: int
    certificate: DriftCertificate
    solver: str = "composition"
    residual_reduction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Claim: infrastructure.
        """
        return {
            "candidate": self.candidate.to_dict(),
            "new_id": self.new_id,
            "certificate": self.certificate.to_dict(),
            "solver": self.solver,
            "residual_reduction": self.residual_reduction,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PackEntry":
        """Inverse of :meth:`to_dict`.

        Claim: infrastructure.
        """
        return cls(
            candidate=MergeCandidate.from_dict(d["candidate"]),
            new_id=int(d["new_id"]),
            certificate=DriftCertificate.from_dict(d["certificate"]),
            solver=d.get("solver", "composition"),
            residual_reduction=float(d.get("residual_reduction", 0.0)),
        )


@dataclass
class VocabPack:
    """A language pack: new tokens + embedding rows + certificates.

    A pack is *append-only with respect to the base model*: it never rewrites an
    existing embedding row.  That single property is what makes the English
    non-regression claim exact rather than statistical — see
    :func:`parity.serving.multi_tokenizer.MultiTokenizerRouter.base_view` and
    ``tests/test_english_nonregression.py``.

    Claim: non-regression, bound, reduction — a pack is the deliverable that
    carries all three.
    """

    lang: str
    base_model_id: str
    base_tokenizer_fingerprint: str
    base_vocab_size: int
    entries: List[PackEntry] = field(default_factory=list)
    input_embeddings: Any = None  # torch.Tensor [n_new, d_model]
    output_embeddings: Any = None  # torch.Tensor [n_new, d_model] or None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def new_ids(self) -> List[int]:
        """Ids assigned to this pack's tokens, in row order.

        Claim: infrastructure.
        """
        return [e.new_id for e in self.entries]

    @property
    def expansions(self) -> Dict[int, Tuple[int, ...]]:
        """Map ``new_id -> base id sequence`` used by encode/decode and serving.

        Claim: infrastructure — the inverse of this map is what guarantees
        lossless round-tripping to the base model's id space.
        """
        return {e.new_id: e.candidate.ids for e in self.entries}

    def worst_bound(self, statistic: str = "kl_next_token") -> float:
        """Largest certified bound over the pack — the number for the card.

        Claim: bound — a pack is only as safe as its worst token, so this is
        the figure the model card must lead with.
        """
        vals = [e.certificate.bounds[statistic].value for e in self.entries if statistic in e.certificate.bounds]
        return max(vals) if vals else 0.0

    def manifest(self) -> Dict[str, Any]:
        """The JSON half of the on-disk format.

        Claim: infrastructure.
        """
        return {
            "format": "parity-vocab-pack/1",
            "lang": self.lang,
            "base_model_id": self.base_model_id,
            "base_tokenizer_fingerprint": self.base_tokenizer_fingerprint,
            "base_vocab_size": self.base_vocab_size,
            "n_tokens": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
            "metadata": self.metadata,
        }


def fingerprint(obj: Any) -> str:
    """Stable SHA-256 over any JSON-serialisable object.

    Used to bind a certificate to the exact calibration data it was computed on
    and a pack to the exact tokenizer it was mined against, so that a mismatch
    is detected at load time instead of silently producing garbage.

    Claim: bound — a guarantee that cannot be traced to its inputs is a slogan.
    """
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def tokenizer_fingerprint(vocab_size: int, probe_encodings: Sequence[Sequence[int]]) -> str:
    """Fingerprint a tokenizer by its size plus its output on fixed probes.

    We deliberately avoid hashing tokenizer files: two serialisations can differ
    byte-wise while behaving identically, and behaviour is what a pack depends
    on.

    Claim: infrastructure.
    """
    return fingerprint({"vocab_size": vocab_size, "probes": [list(p) for p in probe_encodings]})
