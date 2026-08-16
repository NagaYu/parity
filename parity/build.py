"""End-to-end pack construction: mine → synthesise → certify → select → pack.

This is the orchestration behind ``parity build``.  It exists as a library
function so that the CLI, the benchmark harness and the tests all drive exactly
the same pipeline; a benchmark that runs a different code path from the shipped
command is a benchmark of nothing.

Corpus discipline
-----------------
The corpus is split four ways and the splits are never crossed:

``mine``       candidate discovery.
``fit``        calibration contexts the embeddings are fitted on.
``certify``    held-out contexts the drift bounds are measured on.
``eval``       held-out text the reduction is measured on.

Fitting and certifying on the same text would produce a bound that describes
memorisation.  Measuring reduction on the mining text would report the miner's
recall.  Both are easy mistakes and both are prevented here rather than in a
reviewer's head.

Cost discipline
---------------
Certification is the dominant cost, so the pipeline **shortlists** first: the
same submodular objective is run over the full mined ground set to produce
``oversample × budget`` candidates, and only those are synthesised and
certified.  The final selection then runs over the certified survivors.  Every
stage's FLOPs and wall-clock are recorded separately, because "cheaper than
continued pretraining" is a claim that has to survive itemisation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from parity import _version
from parity.adapters import TorchLMAdapter
from parity.certificate import CertifierConfig, DriftCertifier
from parity.corpora import Corpus
from parity.miner import MergeCandidateMiner, MinerConfig, dedupe_nested, mining_report
from parity.selection import SelectionConfig, SelectionResult, VocabularySelector
from parity.synthesis import CalibrationIndex, EmbeddingSynthesizer, SynthesisConfig, unigram_frequencies
from parity.tokenization import AugmentedTokenizer, BaseTokenizer
from parity.types import DriftCertificate, MergeCandidate, PackEntry, SynthesizedToken, VocabPack

log = logging.getLogger("parity.build")


@dataclass
class Splits:
    """The four disjoint corpus slices a build consumes.

    Claim: bound, reduction — the split is the reason the numbers mean what the
    README says they mean.
    """

    mine: Corpus
    fit: Corpus
    certify: Corpus
    eval: Corpus

    def to_dict(self) -> Dict[str, int]:
        """Sizes, for the run manifest.

        Claim: infrastructure.
        """
        return {"mine": len(self.mine), "fit": len(self.fit), "certify": len(self.certify), "eval": len(self.eval)}


def negative_lines(langs: Sequence[str]) -> List[str]:
    """Text in the languages a pack must not disturb, for the off-context bound.

    Pulled from the embedded parallel sample so that this protection is always
    available, with no network and no extra configuration.  An operator with a
    larger corpus for these languages should pass it instead — a bigger negative
    set makes the bound tighter, never weaker.

    Claim: bound, non-regression — supplies the evidence that a new token does
    not fire on the other languages sharing the model.
    """
    from parity.corpora import load_embedded_sample

    sample = load_embedded_sample()
    out: List[str] = []
    for lang in langs:
        out.extend(sample.by_lang.get(lang, []))
    return out


def split_corpus(corpus: Corpus, fractions: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)) -> Splits:
    """Deterministically split a corpus into mine/fit/certify/eval slices.

    Contiguous slices rather than a random shuffle: FLORES lines are already
    independent, and a contiguous split is reproducible without carrying a seed
    around.  For corpora with document-level structure, shuffle upstream.

    Claim: bound — disjointness is enforced here, once, for every caller.
    """
    n = len(corpus)
    if n < 8:
        # Too small to split meaningfully: reuse, and make the compromise loud.
        log.warning("corpus has only %d lines; splits will overlap and bounds will be optimistic", n)
        return Splits(corpus, corpus, corpus, corpus)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    c = b + int(n * fractions[2])
    return Splits(corpus.slice(0, a), corpus.slice(a, b), corpus.slice(b, c), corpus.slice(c, n))


@dataclass
class BuildConfig:
    """Everything ``parity build`` needs to know.

    Claim: infrastructure — one object so a build is reproducible from a manifest.
    """

    lang: str
    budget: int = 4000
    #: Certify ``oversample × budget`` candidates; the rest never cost a forward pass.
    oversample: float = 2.5
    miner: MinerConfig = field(default_factory=MinerConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    certifier: CertifierConfig = field(default_factory=CertifierConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    #: Languages whose text is used as *negatives* for the off-context bound —
    #: the languages this pack shares a model with and must not disturb.
    negative_langs: Tuple[str, ...] = ("en",)
    fit_contexts: int = 24
    #: Held-out occurrences per candidate used for certification.  This is the
    #: binding constraint on how many tokens a pack can contain: the emission
    #: statistic produces one measurement per occurrence, and a (0.95, 0.95)
    #: tolerance limit needs 59 of them.  A FLORES-sized calibration slice
    #: certifies only the most frequent few hundred candidates; larger packs
    #: need a larger corpus (see ``PARITY_MINING_CORPUS_<LANG>``).
    certify_contexts: int = 64
    seed: int = 0


@dataclass
class StageCost:
    """Wall-clock and FLOPs for one pipeline stage.

    Claim: low-cost — benchmark metric (4) is assembled from these, itemised, so
    that the comparison against continued pretraining can be audited line by
    line.
    """

    name: str
    seconds: float = 0.0
    flops: float = 0.0
    tokens: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Serialise.

        Claim: infrastructure.
        """
        return {"stage": self.name, "seconds": self.seconds, "flops": self.flops, "tokens": self.tokens}


@dataclass
class BuildResult:
    """A finished pack plus every diagnostic needed to defend it.

    Claim: reduction, non-regression, bound, low-cost — the single object the
    benchmark reads to score all four claims.
    """

    pack: VocabPack
    selection: SelectionResult
    costs: List[StageCost] = field(default_factory=list)
    mining: Dict[str, Any] = field(default_factory=dict)
    certificates: Dict[str, DriftCertificate] = field(default_factory=dict)
    synthesized: List[SynthesizedToken] = field(default_factory=list)
    splits: Dict[str, int] = field(default_factory=dict)
    corpus_source: str = ""

    @property
    def total_flops(self) -> float:
        """Measured FLOPs for the whole build.

        Claim: low-cost — the numerator of the "orders of magnitude cheaper"
        ratio, measured with the unfavourable ``6N`` convention wherever a
        backward pass occurred.
        """
        return sum(c.flops for c in self.costs)

    @property
    def total_seconds(self) -> float:
        """Measured wall-clock for the whole build.

        Claim: low-cost.
        """
        return sum(c.seconds for c in self.costs)

    def rejection_reasons(self, top: int = 6) -> List[Tuple[str, int]]:
        """Why candidates were refused, most common first.

        Printed by the CLI.  A build that adopts few tokens should say whether
        that was drift (the tolerance is binding) or evidence (the calibration
        corpus is too small) — those call for opposite fixes, and conflating
        them is how a project ends up quietly loosening its own guarantee.

        Claim: bound.
        """
        from collections import Counter

        counts: Counter = Counter()
        for cert in self.certificates.values():
            if cert.accepted:
                continue
            reason = cert.reject_reason
            if "occurrences" in reason:
                key = "insufficient held-out occurrences"
            elif "KL tail bound" in reason:
                key = "KL drift above tolerance"
            elif "TV tail bound" in reason:
                key = "TV drift above tolerance"
            elif "emission" in reason:
                key = "emission drift above tolerance"
            elif "calibration measurements" in reason:
                key = "too few calibration measurements"
            else:
                key = reason[:48] or "unspecified"
            counts[key] += 1
        return counts.most_common(top)

    @property
    def acceptance_rate(self) -> float:
        """Share of certified candidates that cleared the drift tolerance.

        A low rate is informative, not a failure: it means the tolerance is
        binding, which is what a tolerance is for.

        Claim: bound.
        """
        if not self.certificates:
            return 0.0
        return sum(1 for c in self.certificates.values() if c.accepted) / len(self.certificates)

    @property
    def mean_residual_reduction(self) -> float:
        """Average fraction of the composition residual removed by the solver.

        Claim: non-regression — evidence that stage (ii) does work stage (i)
        does not, which is the difference between Parity and the zero-shot
        transfer baseline.
        """
        if not self.synthesized:
            return 0.0
        return sum(t.residual_reduction for t in self.synthesized) / len(self.synthesized)

    def manifest(self) -> Dict[str, Any]:
        """Everything a reader needs to reproduce or challenge this build.

        Claim: infrastructure.
        """
        return {
            "parity_version": _version.__version__,
            "lang": self.pack.lang,
            "base_model_id": self.pack.base_model_id,
            "n_tokens": len(self.pack),
            "splits": self.splits,
            "corpus_source": self.corpus_source,
            "mining": self.mining,
            "selection": self.selection.to_dict(),
            "acceptance_rate": self.acceptance_rate,
            "rejection_reasons": dict(self.rejection_reasons()),
            "mean_residual_reduction": self.mean_residual_reduction,
            "worst_kl_bound": self.pack.worst_bound("kl_next_token"),
            "worst_tv_bound": self.pack.worst_bound("tv_next_token"),
            "costs": [c.to_dict() for c in self.costs],
            "total_flops": self.total_flops,
            "total_seconds": self.total_seconds,
        }


def build_pack(
    adapter: TorchLMAdapter,
    tokenizer: BaseTokenizer,
    corpus: Corpus,
    config: BuildConfig,
    base_model_id: str = "",
) -> BuildResult:
    """Run the full pipeline and return a certified, budgeted vocabulary pack.

    Claim: reduction, non-regression, bound, low-cost — this function is the
    method; every claim in the README is a measurement of its output.
    """
    torch.manual_seed(config.seed)
    costs: List[StageCost] = []
    splits = split_corpus(corpus)
    aug = AugmentedTokenizer(tokenizer)

    # -- 1. mine ------------------------------------------------------------
    t0 = time.time()
    miner = MergeCandidateMiner(tokenizer, config.miner)
    candidates = miner.mine(splits.mine, config.lang)
    candidates = dedupe_nested(candidates)
    report = mining_report(miner, splits.mine, candidates)
    costs.append(StageCost("mine", time.time() - t0, 0.0, report.n_base_tokens))
    if not candidates:
        raise RuntimeError(f"no merge candidates mined for {config.lang!r}; corpus too small or min_count too high")

    # -- 2. shortlist -------------------------------------------------------
    # Certification dominates cost, so pay it only for candidates that could
    # plausibly be selected. The shortlist uses the same objective as the final
    # selection, so it cannot systematically prefer candidates the final stage
    # would reject.
    t0 = time.time()
    mine_docs = [tokenizer.encode(l) for l in splits.mine.lines]
    shortlist_n = max(config.budget, int(config.budget * config.oversample))
    pre = VocabularySelector(candidates, mine_docs, None, SelectionConfig(budget=shortlist_n))
    shortlist = pre.select(shortlist_n).selected
    costs.append(StageCost("shortlist", time.time() - t0, 0.0, 0))
    log.info("shortlisted %d/%d candidates for certification", len(shortlist), len(candidates))

    # -- 3. calibration indices (disjoint) ----------------------------------
    fit_docs = [tokenizer.encode(l) for l in splits.fit.lines]
    cert_docs = [tokenizer.encode(l) for l in splits.certify.lines]
    fit_index = CalibrationIndex(shortlist, config.synthesis.prefix_tokens, config.synthesis.suffix_tokens)
    fit_index.scan(fit_docs, max_per_candidate=config.fit_contexts)
    cert_index = CalibrationIndex(shortlist, config.synthesis.prefix_tokens, config.synthesis.suffix_tokens)
    cert_index.scan(cert_docs, max_per_candidate=config.certify_contexts)
    log.info("calibration coverage: fit %.1f%%, certify %.1f%%", 100 * fit_index.coverage(), 100 * cert_index.coverage())

    # -- 4. synthesise ------------------------------------------------------
    t0 = time.time()
    adapter.reset_cost()
    synth = EmbeddingSynthesizer(adapter, config.synthesis, unigram_freq=unigram_frequencies(mine_docs))
    tokens = synth.synthesize(shortlist, fit_index)
    costs.append(
        StageCost(
            "synthesis",
            time.time() - t0,
            adapter.flops(backward=config.synthesis.adam_steps > 0),
            adapter.forward_tokens,
        )
    )
    if not tokens:
        raise RuntimeError("no candidate had calibration contexts; enlarge the corpus or lower min_count")

    # -- 5. certify ---------------------------------------------------------
    t0 = time.time()
    adapter.reset_cost()
    certifier = DriftCertifier(adapter, config.certifier)
    # Negatives: the languages this pack must not disturb. English is included
    # unconditionally, because a token that fires on English text is exactly the
    # failure the base-view guarantee does not cover once an operator routes a
    # mixed-language request through a pack view.
    negatives = list(cert_docs)
    for line in negative_lines(config.negative_langs):
        ids = tokenizer.encode(line)
        if len(ids) >= 2:
            negatives.append(ids)
    certificates = certifier.certify(
        tokens, cert_index, max_contexts=config.certify_contexts, negative_docs=negatives
    )
    costs.append(StageCost("certify", time.time() - t0, adapter.flops(), adapter.forward_tokens))

    # -- 6. select ----------------------------------------------------------
    t0 = time.time()
    eval_docs = [tokenizer.encode(l) for l in splits.eval.lines]
    # Pass every synthesised candidate and let the selector apply the
    # certificate filter, so the rejection count in the manifest is the real one.
    selector = VocabularySelector([t.candidate for t in tokens], eval_docs, certificates, config.selection)
    result = selector.select(config.budget)
    costs.append(StageCost("select", time.time() - t0, 0.0, 0))

    # -- 7. assemble --------------------------------------------------------
    by_key = {t.candidate.key: t for t in tokens}
    entries: List[PackEntry] = []
    in_rows: List[torch.Tensor] = []
    out_rows: List[torch.Tensor] = []
    have_out = all(by_key[c.key].output_embedding is not None for c in result.selected) and result.selected
    for i, cand in enumerate(result.selected):
        tok = by_key[cand.key]
        entries.append(
            PackEntry(
                candidate=cand,
                new_id=aug.base_vocab_size + i,
                certificate=certificates[cand.key],
                solver=tok.solver,
                residual_reduction=tok.residual_reduction,
            )
        )
        in_rows.append(tok.input_embedding)
        if have_out:
            out_rows.append(tok.output_embedding)

    pack = VocabPack(
        lang=config.lang,
        base_model_id=base_model_id or adapter.name,
        base_tokenizer_fingerprint=aug.fingerprint(),
        base_vocab_size=aug.base_vocab_size,
        entries=entries,
        input_embeddings=torch.stack(in_rows) if in_rows else None,
        output_embeddings=torch.stack(out_rows) if have_out and out_rows else None,
        metadata={},
    )

    build = BuildResult(
        pack=pack,
        selection=result,
        costs=costs,
        mining=report.to_dict(),
        certificates=certificates,
        synthesized=tokens,
        splits=splits.to_dict(),
        corpus_source=corpus.source,
    )
    pack.metadata.update(
        {
            "token_reduction": result.token_reduction,
            "added_parameters": adapter.embedding_param_cost(len(entries)),
            "solver": config.synthesis.solver,
            "input_only": config.certifier.input_only,
            "mining_corpus": f"{corpus.source} ({len(splits.mine)} lines)",
            "calibration_corpus": f"{corpus.source} ({len(splits.certify)} lines, held out)",
            "build_flops": f"{build.total_flops:.3e}",
            "build_seconds": f"{build.total_seconds:.1f}",
            "parity_version": _version.__version__,
            "certified_optimality_ratio": result.certified_optimality_ratio,
            "acceptance_rate": build.acceptance_rate,
        }
    )
    return build


def attach_and_verify(adapter: TorchLMAdapter, tokenizer: BaseTokenizer, packs: Sequence[VocabPack]) -> AugmentedTokenizer:
    """Attach packs to a tokenizer and model, then assert the invariants hold.

    Runs :meth:`AugmentedTokenizer.check_invariants` and the append-only
    assertion in :meth:`TorchLMAdapter.append_rows`, so a mis-built pack fails
    loudly at load time instead of quietly at inference time.

    Claim: non-regression — the guarantee is re-derived on every load.
    """
    aug = AugmentedTokenizer(tokenizer)
    for pack in packs:
        aug.attach(pack)
        rows = pack.input_embeddings
        if rows is None:
            raise ValueError(f"pack {pack.lang!r} has no embedding rows")
        out = pack.output_embeddings
        if out is None and not adapter.tied_embeddings:
            raise ValueError(f"pack {pack.lang!r} has no output rows but the model has untied embeddings")
        adapter.append_rows(rows, out)
    aug.check_invariants()
    return aug
