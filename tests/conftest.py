"""Shared fixtures.

Everything here runs offline in seconds against :class:`~parity.tiny.TinyCausalLM`
and a BPE trained on the embedded sample.  That is deliberate: Parity's central
guarantees are *structural* — append-only rows, view isolation, disjoint
calibration splits, a correctly computed order statistic — and a structural
guarantee that can only be checked by downloading a billion parameters will not
be checked.

Two honest caveats about the fixture, restated wherever they matter:

* The fixture model is **untrained**, so its residual-stream geometry is noisier
  than a real checkpoint's.  Absolute drift magnitudes here mean nothing; the
  tests assert relations (bound holds, base view identical, reduction positive),
  never magnitudes.
* :func:`~parity.corpora.expand_for_testing` builds lines by pairing 24 sample
  sentences, so different splits share vocabulary heavily.  Splits are still
  line-disjoint, but the calibration distribution shift between them is smaller
  than in real data, which makes the certificate test *easier* than reality.
  ``tests/test_certificate.py`` says so where it asserts.
"""

from __future__ import annotations

import pytest
import torch

from parity.adapters import TorchLMAdapter
from parity.build import BuildConfig, build_pack
from parity.certificate import CertifierConfig
from parity.corpora import Corpus, expand_for_testing, load_embedded_sample
from parity.miner import MinerConfig
from parity.selection import SelectionConfig
from parity.synthesis import SynthesisConfig
from parity.tiny import build_tiny_model, build_tiny_tokenizer
from parity.tokenization import AugmentedTokenizer

TARGET = "ja"


@pytest.fixture(scope="session")
def sample():
    """The embedded 24-sentence parallel corpus."""
    return load_embedded_sample()


@pytest.fixture(scope="session")
def base_tokenizer(sample):
    """A BPE fit on English only — the asymmetry Parity exists to repair.

    English-only on purpose.  Including a few target-language sentences would
    let the BPE memorise those exact sentences, which would then tokenize to one
    or two tokens and make every downstream test vacuous.  Training on English
    alone reproduces the real situation in its clean form: other scripts fall
    back toward byte level.
    """
    return build_tiny_tokenizer(sample.by_lang["en"] * 8, vocab_size=900)


@pytest.fixture(scope="session")
def corpus(sample):
    """A line-disjoint-splittable fixture corpus for the target language."""
    return Corpus(TARGET, expand_for_testing(sample.by_lang[TARGET], 552, seed=1), source="fixture:expanded_sample")


def _make_adapter(base_tokenizer, tie: bool, seed: int = 0) -> TorchLMAdapter:
    model = build_tiny_model(base_tokenizer.vocab_size, seed=seed, tie=tie)
    return TorchLMAdapter(model, name="tiny-tied" if tie else "tiny")


@pytest.fixture()
def adapter(base_tokenizer):
    """A fresh untied fixture model (fresh, because packs mutate the matrix)."""
    return _make_adapter(base_tokenizer, tie=False)


@pytest.fixture()
def tied_adapter(base_tokenizer):
    """A fresh fixture model with tied embeddings — the common real-world case."""
    return _make_adapter(base_tokenizer, tie=True)


def demo_build_config(budget: int = 48) -> BuildConfig:
    """Build settings tuned for a two-second offline run.

    Tolerances are loose because the fixture model is untrained; the tests never
    assert on their magnitude, only that measured drift respects whatever bound
    was issued.
    """
    return BuildConfig(
        lang=TARGET,
        budget=budget,
        oversample=2.0,
        miner=MinerConfig(min_count=3, min_doc_count=2, max_length=6),
        synthesis=SynthesisConfig(subspace_dim=6, gn_iters=2, max_contexts=6, chunk_size=16, output_contexts=24),
        certifier=CertifierConfig(
            max_kl=2.0,
            max_tv=0.6,
            # The fixture model is untrained, so the log-probability of a
            # merge's continuation is ~-27 nats instead of the ~-0.5 a trained
            # model gives a good candidate. No output row inside the trust
            # region can reach that, so the emission tolerance here is set
            # where it does not bind; the tests assert the *bound holds*, never
            # that it is small.
            max_emit_logprob_err=40.0,
            max_offcontext_mass=0.05,
            # (0.80, 0.80) rather than the shipped (0.95, 0.95): the fixture's
            # certify split is ~110 short lines, and a 95/95 tolerance limit
            # needs 59 held-out occurrences of every token. Weakening the
            # guarantee here is honest; silently accepting tokens without the
            # evidence for it would not be.
            alpha=0.2,
            delta=0.2,
            min_calibration=8,
        ),
        selection=SelectionConfig(budget=budget),
        certify_contexts=32,
        fit_contexts=24,
    )


@pytest.fixture(scope="session")
def built(base_tokenizer, corpus):
    """A complete build (untied model) shared by the read-only tests.

    Session-scoped because it is the expensive fixture; tests that mutate the
    model (attaching a pack appends rows) build their own adapter and re-attach.
    """
    adapter = _make_adapter(base_tokenizer, tie=False)
    result = build_pack(adapter, base_tokenizer, corpus, demo_build_config(), base_model_id="tiny")
    return {"result": result, "adapter": adapter, "tokenizer": base_tokenizer, "corpus": corpus}


@pytest.fixture()
def attached(built, base_tokenizer):
    """A fresh model+tokenizer with the session pack attached and verified."""
    from parity.build import attach_and_verify

    adapter = _make_adapter(base_tokenizer, tie=False)
    pack = built["result"].pack
    aug = attach_and_verify(adapter, base_tokenizer, [pack])
    return {"adapter": adapter, "tokenizer": aug, "pack": pack}


@pytest.fixture()
def router(attached):
    """A :class:`~parity.serving.MultiTokenizerRouter` over the attached pack."""
    from parity.serving import MultiTokenizerRouter

    return MultiTokenizerRouter(attached["adapter"], attached["tokenizer"])


@pytest.fixture()
def plain_augmented(base_tokenizer):
    """An :class:`AugmentedTokenizer` with no packs attached."""
    return AugmentedTokenizer(base_tokenizer)
