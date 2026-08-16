"""Pack I/O, the model card, and the command line.

The model card test matters more than it looks: the card is what a downstream
user reads instead of the manifest, so a card that could disagree with the
evidence is a supply-chain problem, not a formatting one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from parity.pack import certificate_table, load_pack, model_card, save_pack


def test_pack_round_trips_through_disk(built, tmp_path):
    pack = built["result"].pack
    out = save_pack(pack, tmp_path / "ja")
    assert (out / "manifest.json").exists()
    assert (out / "embeddings.safetensors").exists()
    assert (out / "README.md").exists()
    assert (out / "added_tokens.json").exists()

    loaded = load_pack(out)
    assert len(loaded) == len(pack)
    assert loaded.lang == pack.lang
    assert loaded.base_vocab_size == pack.base_vocab_size
    assert loaded.base_tokenizer_fingerprint == pack.base_tokenizer_fingerprint
    assert torch.allclose(loaded.input_embeddings, pack.input_embeddings)
    for a, b in zip(loaded.entries, pack.entries):
        assert a.candidate.ids == b.candidate.ids
        assert a.certificate.value("kl_next_token") == pytest.approx(b.certificate.value("kl_next_token"))


def test_load_pack_drops_unaccepted_entries(built, tmp_path):
    import copy

    pack = copy.deepcopy(built["result"].pack)
    n = len(pack)
    assert n >= 2
    object.__setattr__(pack.entries[0].certificate, "accepted", False)
    save_pack(pack, tmp_path / "mixed")
    loaded = load_pack(tmp_path / "mixed", require_accepted=True)
    assert len(loaded) == n - 1
    assert loaded.input_embeddings.shape[0] == n - 1, "embedding rows must be dropped with their entries"
    assert len(load_pack(tmp_path / "mixed", require_accepted=False)) == n


def test_model_card_states_the_certified_bound_from_the_manifest(built):
    pack = built["result"].pack
    pack.metadata.update({"token_reduction": built["result"].selection.token_reduction})
    card = model_card(pack)
    worst = pack.worst_bound("kl_next_token")
    assert f"{worst:.4g}" in card, "the card must quote the manifest's own worst-case bound"
    assert "license: apache-2.0" in card and "base_model:" in card
    assert "not* worst-case over all possible inputs" in card or "not worst-case" in card.replace("*", "")
    # The framing commitment must survive into the public artefact.
    assert "property of the base tokenizer" in card
    assert "It says nothing about the language." in card


def test_certificate_table_lists_the_worst_tokens_first(built):
    table = certificate_table(built["result"].pack, n=5)
    lines = [l for l in table.splitlines() if l.startswith("| `")]
    values = [float(l.split("|")[3].strip()) for l in lines]
    assert values == sorted(values, reverse=True)


def test_cli_demo_runs_end_to_end(tmp_path, capsys):
    from parity.cli import main

    rc = main(["demo", "--lang", "ja", "--budget", "24", "--out", str(tmp_path / "demo"), "--offline"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "token reduction" in out
    assert "certified KL bound" in out
    assert "English" in out
    manifest = json.loads((tmp_path / "demo" / "build_manifest.json").read_text())
    assert manifest["n_tokens"] >= 1
    assert manifest["worst_kl_bound"] >= 0


def test_cli_inspect_prints_surfaces_not_ids(tmp_path, capsys):
    from parity.cli import main

    main(["demo", "--lang", "ja", "--budget", "16", "--out", str(tmp_path / "d"), "--offline"])
    capsys.readouterr()
    rc = main(["inspect", "--pack", str(tmp_path / "d"), "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "surface" in out and "KL bound" in out


def test_cli_atlas_prints_a_parity_ratio_column(capsys):
    from parity.cli import main

    rc = main(["atlas", "--model", "tiny", "--langs", "ja,hi,ar", "--offline", "--fixture-corpus"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vs English" in out and "eff. context" in out
    assert "ja" in out and "hi" in out


def test_cli_verify_passes_on_a_freshly_built_pack(tmp_path, capsys):
    from parity.cli import main

    main(["demo", "--lang", "ja", "--budget", "16", "--out", str(tmp_path / "p"), "--offline"])
    capsys.readouterr()
    rc = main(
        [
            "verify",
            "--model", "tiny",
            "--pack", str(tmp_path / "p"),
            "--offline",
            "--fixture-corpus",
            "--alpha", "0.15",
            "--delta", "0.1",
        ]
    )
    out = capsys.readouterr().out
    assert "mean coverage" in out and "RESULT: PASS" in out
    assert rc == 0, out


def test_fingerprint_is_behavioural_not_file_based(base_tokenizer):
    from parity.tokenization import AugmentedTokenizer

    a = AugmentedTokenizer(base_tokenizer).fingerprint()
    b = AugmentedTokenizer(base_tokenizer).fingerprint()
    assert a == b and len(a) == 64
