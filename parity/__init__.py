"""Parity — certified vocabulary augmentation without continued pretraining.

Parity adds language-specific tokens to an already-trained open-weight model so
that text in under-served languages costs fewer tokens, while proving (to a
stated confidence level) that the model's behaviour does not move more than a
declared amount.

The package is organised around five components, each of which carries evidence
for exactly one of the project's four claims:

===================  ===================================================
Component            Claim it produces evidence for
===================  ===================================================
``miner``            reduction  — where the tokens are being wasted
``synthesis``        non-regression, low-cost — new rows without training
``certificate``      bound      — how far behaviour can move
``selection``        reduction  — most saving per unit of embedding budget
``serving``          non-regression, low-cost — one model, many views
===================  ===================================================

Every public function's docstring ends with a ``Claim:`` line naming which of
``reduction`` / ``non-regression`` / ``bound`` / ``low-cost`` /
``infrastructure`` it substantiates.  ``tests/test_docstring_claims.py``
enforces this.

A note on framing
-----------------
Nothing here treats a language as "inefficient".  Token cost is a property of a
*tokenizer* — an artefact built from a training corpus that under-represented
most of the world's writing systems.  Parity is a repair for that artefact.
See ``docs/framing.md``.
"""

from parity._version import __version__

__all__ = [
    "__version__",
    "MergeCandidate",
    "SynthesizedToken",
    "DriftCertificate",
    "VocabPack",
    "FertilityReport",
]


def __getattr__(name: str):  # pragma: no cover - lazy re-export
    """Lazily re-export the core dataclasses so ``import parity`` stays cheap.

    Claim: infrastructure — keeps the Gradio Space cold-start fast by not
    importing torch until a component that needs it is touched.
    """
    if name in {"MergeCandidate", "SynthesizedToken", "DriftCertificate", "VocabPack", "FertilityReport"}:
        from parity import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module 'parity' has no attribute {name!r}")
