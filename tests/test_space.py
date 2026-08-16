"""The published static Space: assets present, in sync, and honestly worded.

The Space is the artefact most people will ever see. These tests keep it from
drifting away from the repository in the two ways that would matter: quoting
data the repository no longer produces, and quoting a claim the repository does
not support.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


@pytest.fixture(scope="module")
def index_html() -> str:
    path = SPACE / "index.html"
    assert path.exists(), "space/index.html is the published Space; it must be in the repo"
    return path.read_text(encoding="utf-8")


def test_every_asset_the_page_fetches_exists(index_html):
    fetched = set(re.findall(r'fetch\("([^"]+)"\)', index_html))
    local = {f for f in fetched if not f.startswith("http")}
    assert local, "the page should load its data from local JSON, not hardcode it"
    for name in local:
        assert (SPACE / name).exists(), f"index.html fetches {name}, which is not in space/"


def test_space_data_is_in_sync_with_the_repository():
    from scripts.build_space_assets import atlas_payload, sample_payload

    assert json.loads((SPACE / "sample.json").read_text(encoding="utf-8")) == sample_payload(), (
        "space/sample.json has drifted; run scripts/build_space_assets.py"
    )
    atlas = json.loads((SPACE / "atlas.json").read_text(encoding="utf-8"))
    expected = atlas_payload()
    if expected:
        assert atlas == expected, "space/atlas.json has drifted; run scripts/build_space_assets.py"


def test_atlas_rows_carry_the_fields_the_page_reads(index_html):
    rows = json.loads((SPACE / "atlas.json").read_text(encoding="utf-8"))
    if not rows:
        pytest.skip("atlas not built in this checkout")
    for field in ("tokenizer_id", "lang", "parity_ratio"):
        assert field in rows[0], f"the page reads r.{field}"
        assert f".{field}" in index_html or f'"{field}"' in index_html


def test_space_states_the_framing_commitment(index_html):
    # docs/framing.md is a constraint on every string this project emits, and the
    # Space is where most readers meet the project.
    assert "property of the tokenizer, not of your language" in index_html
    assert "not a property of any language" in index_html
    lowered = index_html.lower()
    for banned in ("inefficient language", "verbose language", "exotic script", "hard language"):
        assert banned not in lowered, f"framing violation in the Space: {banned!r}"


def test_space_states_the_scope_limit(index_html):
    assert "closed API" in index_html, "the Space must say where Parity cannot be applied"
    assert "open-weight" in index_html


def test_space_does_not_overstate_the_certificate(index_html):
    # The bound is finite-sample and distribution-free, not worst-case. If the
    # page ever claims a guarantee, it must claim the right one.
    if "certified bound" in index_html or "certificate" in index_html.lower():
        assert "finite-sample" in index_html and "distribution-free" in index_html
        assert "not worst-case" in index_html


def test_merge_pass_is_leftmost_longest_in_the_page_too(index_html):
    # The browser reimplements one thing from the Python: the merge pass. If it
    # diverged, the token counts shown would not be the ones a pack delivers.
    assert "leftmost-longest" in index_html
    assert "bestLen" in index_html and "trie" in index_html
