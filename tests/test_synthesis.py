"""Synthesis: the least-squares stage must do real work, and touch no weights.

The claim being tested is not "the embeddings are good" — the certificate
handles that — but the two structural properties that make Parity what it says
it is:

* it is **not** continued pretraining: no model parameter changes, and no
  parameter is even reachable from the objective;
* stage (ii) improves measurably on stage (i), otherwise Parity would just be
  the zero-shot transfer baseline with extra steps.
"""

from __future__ import annotations

import pytest
import torch

from parity.baselines import zero_shot_transfer_config
from parity.corpora import expand_for_testing
from parity.synthesis import CalibrationIndex, EmbeddingSynthesizer, SynthesisConfig, unigram_frequencies


def _index(candidates, docs, n=8):
    idx = CalibrationIndex(candidates, 12, 6)
    idx.scan(docs, max_per_candidate=n)
    return idx


def test_no_model_parameter_changes_during_synthesis(adapter, base_tokenizer, corpus, built):
    before = {name: p.detach().clone() for name, p in adapter.model.named_parameters()}
    cands = [e.candidate for e in built["result"].pack.entries][:6]
    if not cands:
        pytest.skip("fixture pack empty")
    docs = [base_tokenizer.encode(l) for l in corpus.lines[:120]]
    synth = EmbeddingSynthesizer(adapter, SynthesisConfig(subspace_dim=4, gn_iters=2, max_contexts=6))
    synth.synthesize(cands, _index(cands, docs))
    for name, p in adapter.model.named_parameters():
        assert torch.equal(before[name], p.detach()), f"synthesis modified model parameter {name}"


def test_all_parameters_are_frozen_by_the_adapter(adapter):
    assert all(not p.requires_grad for p in adapter.model.parameters())


def test_least_squares_improves_on_the_composition_baseline(adapter, base_tokenizer, corpus, built):
    cands = [e.candidate for e in built["result"].pack.entries][:8]
    if not cands:
        pytest.skip("fixture pack empty")
    docs = [base_tokenizer.encode(l) for l in corpus.lines[:160]]
    idx = _index(cands, docs)
    freq = unigram_frequencies(docs)

    gn = EmbeddingSynthesizer(adapter, SynthesisConfig(subspace_dim=6, gn_iters=2, max_contexts=6), freq)
    tokens = gn.synthesize(cands, idx)
    assert tokens
    # The solver is monotone by construction (it rejects non-improving steps).
    for t in tokens:
        assert t.residual_after <= t.residual_before + 1e-6
    mean_reduction = sum(t.residual_reduction for t in tokens) / len(tokens)
    assert mean_reduction > 0.1, f"the least-squares stage only removed {100 * mean_reduction:.1f}% of the residual"


def test_zero_shot_transfer_config_is_pure_composition(adapter, base_tokenizer, corpus, built):
    cands = [e.candidate for e in built["result"].pack.entries][:4]
    if not cands:
        pytest.skip("fixture pack empty")
    docs = [base_tokenizer.encode(l) for l in corpus.lines[:120]]
    idx = _index(cands, docs)
    cfg = zero_shot_transfer_config()
    synth = EmbeddingSynthesizer(adapter, cfg)
    tokens = synth.synthesize(cands, idx)
    for t, c in zip(tokens, cands):
        assert t.solver == "composition"
        assert t.output_embedding is not None  # untied model: FVT composes one
        expected = adapter.embed_ids(c.ids).float().mean(0)
        assert torch.allclose(t.input_embedding, expected, atol=1e-5)


def test_norm_matching_puts_the_row_on_the_embedding_manifold(adapter, built):
    cands = [e.candidate for e in built["result"].pack.entries][:4]
    if not cands:
        pytest.skip("fixture pack empty")
    plain = EmbeddingSynthesizer(adapter, SynthesisConfig(composition="mean", norm_match=False))
    matched = EmbeddingSynthesizer(adapter, SynthesisConfig(composition="mean", norm_match=True))
    target = matched._mean_embedding_norm()
    for c in cands:
        raw = plain.compose(c.ids)
        fixed = matched.compose(c.ids)
        assert float(raw.norm()) < target, "averaging sub-tokens should shrink the norm"
        assert float(fixed.norm()) == pytest.approx(target, rel=1e-4)


def test_candidates_without_calibration_contexts_are_skipped(adapter, built, base_tokenizer):
    from parity.types import MergeCandidate

    real = [e.candidate for e in built["result"].pack.entries][:2]
    if not real:
        pytest.skip("fixture pack empty")
    impossible = MergeCandidate(ids=(9997, 9998, 9999), surface="never", count=1, lang="ja")
    cands = real + [impossible]
    docs = [base_tokenizer.encode(l) for l in built["corpus"].lines[:80]]
    synth = EmbeddingSynthesizer(adapter, SynthesisConfig(subspace_dim=4, gn_iters=1, max_contexts=4))
    tokens = synth.synthesize(cands, _index(cands, docs))
    assert impossible.key not in {t.candidate.key for t in tokens}


def test_tied_model_folds_emission_into_the_input_objective(tied_adapter, base_tokenizer, built, corpus):
    cands = [e.candidate for e in built["result"].pack.entries][:6]
    if not cands:
        pytest.skip("fixture pack empty")
    docs = [base_tokenizer.encode(l) for l in corpus.lines[:160]]
    idx = _index(cands, docs)
    assert tied_adapter.tied_embeddings

    with_emit = EmbeddingSynthesizer(
        tied_adapter, SynthesisConfig(subspace_dim=6, gn_iters=2, max_contexts=6, emit_weight=1.0)
    ).synthesize(cands, idx)
    without = EmbeddingSynthesizer(
        tied_adapter, SynthesisConfig(subspace_dim=6, gn_iters=2, max_contexts=6, emit_weight=0.0)
    ).synthesize(cands, idx)

    assert all(t.output_embedding is None for t in with_emit), "a tied model has no separate output row"
    # The emission term must actually change the solution, or it is not doing
    # the job the tied case needs it for.
    diffs = [float((a.input_embedding - b.input_embedding).norm()) for a, b in zip(with_emit, without)]
    assert max(diffs) > 1e-4


def test_fitted_output_row_beats_its_own_anchor(adapter, base_tokenizer, built, corpus):
    """The ridge correction must reduce emission error, not merely exist."""
    import copy

    from parity.certificate import DriftCertifier

    cands = [e.candidate for e in built["result"].pack.entries][:4]
    if not cands:
        pytest.skip("fixture pack empty")
    docs = [base_tokenizer.encode(l) for l in corpus.lines[:160]]
    idx = _index(cands, docs, n=24)
    synth = EmbeddingSynthesizer(adapter, SynthesisConfig(subspace_dim=4, gn_iters=1, max_contexts=6))
    tokens = synth.synthesize(cands, idx)
    assert tokens and all(t.output_embedding is not None for t in tokens)

    certifier = DriftCertifier(adapter, None)
    W = adapter.unembed_matrix()
    improved = 0
    for t in tokens:
        ctxs = idx.contexts_for(t.candidate)[:16]
        if len(ctxs) < 4:
            continue
        anchored = copy.copy(t)
        anchored.output_embedding = W[int(t.candidate.ids[0])].float().clone()
        fitted_err = certifier.measure(t, ctxs)["emit_logprob_err"]
        anchor_err = certifier.measure(anchored, ctxs)["emit_logprob_err"]
        if fitted_err and anchor_err:
            improved += int(sum(fitted_err) / len(fitted_err) < sum(anchor_err) / len(anchor_err))
    assert improved >= 1, "the ridge fit never improved on the raw first-sub-token anchor"
