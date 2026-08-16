"""Every number the README quotes must still be the number the data holds.

This project's whole argument is that a card must not drift from its evidence —
`parity.pack.model_card` is generated from the manifest for exactly that reason.
The README is the biggest card of all and was, until this test existed, the one
document written by hand. It had three wrong figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ATLAS = ROOT / "data" / "atlas.jsonl"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def atlas():
    if not ATLAS.exists():
        pytest.skip("atlas not built in this checkout")
    return [json.loads(line) for line in ATLAS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(atlas, lang: str, tokenizer_substring: str):
    return next(
        (r["parity_ratio"] for r in atlas if r["lang"] == lang and tokenizer_substring in r["tokenizer_id"]),
        None,
    )


#: Rows of the README's headline fertility table: (label, ISO code).
TABLE_ROWS = [
    ("Burmese", "my"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Bengali", "bn"),
    ("Hindi", "hi"),
    ("Amharic", "am"),
    ("Thai", "th"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
]


@pytest.mark.parametrize("label,lang", TABLE_ROWS)
def test_readme_fertility_row_matches_the_atlas(readme, atlas, label, lang):
    row = next((l for l in readme.splitlines() if l.startswith(f"| {label}")), None)
    assert row, f"README has no fertility row for {label}"
    cells = [c.strip() for c in row.strip("|").split("|")]
    quoted = [float(m.group(1)) for c in cells for m in [re.search(r"([\d.]+)x", c)] if m]
    assert len(quoted) == 2, f"expected a Qwen and a SmolLM2 ratio in: {row}"

    for got, tok in zip(quoted, ("Qwen", "SmolLM")):
        want = _ratio(atlas, lang, tok)
        assert want is not None, f"atlas has no {lang} row for {tok}"
        assert abs(got - want) < 0.005, (
            f"README says {label} is {got}x under {tok}, atlas says {want:.2f}x — "
            "regenerate the table from data/atlas.jsonl"
        )


def test_readme_does_not_claim_a_reduction_the_pack_does_not_deliver(readme):
    # The published pack adopts 2 of 96 candidates. The README must lead with
    # that, not with the 30% the shortlist would have saved uncertified.
    assert "2 of 96" in readme or "**2 of 96**" in readme
    assert "negative one" in readme or "negative result" in readme


def test_readme_states_the_certificate_is_not_worst_case(readme):
    assert "not worst-case over all inputs" in readme or "not worst-case" in readme
    assert "finite-sample" in readme and "distribution-free" in readme
