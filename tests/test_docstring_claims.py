"""Every public callable must say which claim it substantiates.

The project brief requires each function's docstring to name the claim it
provides evidence for — reduction, non-regression, bound, or low-cost.  A
convention that is not enforced is a convention that decays, so it is enforced
here: this test walks the package and fails on any public callable without a
``Claim:`` line naming a known tag.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import parity

VALID = {"reduction", "non-regression", "bound", "low-cost", "infrastructure"}

MODULES = [
    "parity.adapters",
    "parity.baselines",
    "parity.build",
    "parity.certificate",
    "parity.cli",
    "parity.corpora",
    "parity.fertility",
    "parity.miner",
    "parity.pack",
    "parity.selection",
    "parity.serving.multi_tokenizer",
    "parity.serving.prefix_cache",
    "parity.synthesis",
    "parity.tiny",
    "parity.tokenization",
    "parity.types",
    "benchmarks.tasks",
    "benchmarks.cost",
    "benchmarks.report",
    "benchmarks.run",
    "figures.make_pareto",
    "serving.vllm_plugin",
    "serving.tgi_plugin",
    "scripts.build_atlas",
    "scripts.push_vocab_pack",
]


def _public_callables(module):
    import dataclasses

    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            yield f"{module.__name__}.{name}", obj
        elif inspect.isclass(obj):
            if getattr(obj, "_is_protocol", False):
                # Protocol method stubs have no bodies to document; the protocol
                # class docstring carries the claim.
                yield f"{module.__name__}.{name}", obj
                continue
            yield f"{module.__name__}.{name}", obj
            for mname, meth in vars(obj).items():
                # `__init__` is exempt: the class docstring states the claim, and
                # repeating it on the constructor is noise, not evidence.
                if mname.startswith("_"):
                    continue
                target = meth.fget if isinstance(meth, property) else meth
                if inspect.isfunction(target):
                    yield f"{module.__name__}.{name}.{mname}", target


def _claims(doc: str):
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("Claim:"):
            body = line[len("Claim:") :]
            body = body.split("—")[0].split(" - ")[0]
            return {t.strip().rstrip(".").strip() for t in body.split(",") if t.strip()}
    return None


@pytest.mark.parametrize("module_name", MODULES)
def test_public_callables_declare_a_claim(module_name):
    module = importlib.import_module(module_name)
    missing, bad = [], []
    for qualname, fn in _public_callables(module):
        doc = inspect.getdoc(fn)
        if not doc:
            missing.append(qualname)
            continue
        tags = _claims(doc)
        if tags is None:
            missing.append(qualname)
        elif not tags <= VALID:
            bad.append((qualname, tags - VALID))
    assert not missing, "missing a 'Claim:' line: " + ", ".join(missing)
    assert not bad, "unknown claim tags: " + ", ".join(f"{q}{t}" for q, t in bad)


def test_the_package_docstring_states_the_framing_commitment():
    doc = parity.__doc__ or ""
    assert "tokenizer" in doc.lower()
    assert "inefficient" in doc.lower() or "not a property of the language" in doc.lower() or "artefact" in doc.lower()


def test_every_module_is_importable():
    for _, name, _ in pkgutil.walk_packages(parity.__path__, prefix="parity."):
        importlib.import_module(name)
