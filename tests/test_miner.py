"""Mining: the ground set must be clean, complete and cheap to produce."""

from __future__ import annotations

import pytest

from parity.corpora import Corpus
from parity.miner import MergeCandidateMiner, MinerConfig, dedupe_nested, mining_report
from parity.tokenization import ByteTokenizer
from parity.types import MergeCandidate


class _FakeTok:
    """A tokenizer with a deliberate non-round-tripping surface form."""

    name = "fake"
    vocab_size = 10

    def encode(self, text):
        return [ord(c) - 96 for c in text if c.isalpha()]

    def decode(self, ids):
        return "".join(chr(i + 96) for i in ids)


def test_apriori_finds_frequent_ngrams_and_respects_min_count():
    tok = _FakeTok()
    corpus = Corpus("xx", ["abc def"] * 5 + ["abc xyz"] * 5 + ["qqq"] * 1)
    miner = MergeCandidateMiner(tok, MinerConfig(min_count=3, min_doc_count=2, max_length=4))
    cands = miner.mine(corpus)
    keys = {c.surface for c in cands}
    assert "abc" in keys or "ab" in keys
    for c in cands:
        assert c.count >= 3 and c.doc_count >= 2
        assert c.length >= 2


def test_round_trip_filter_rejects_unencodable_surfaces():
    tok = ByteTokenizer()
    corpus = Corpus("ja", ["子どもたちが公園で遊んでいます。"] * 6)
    miner = MergeCandidateMiner(tok, MinerConfig(min_count=3, min_doc_count=2, max_length=4, require_round_trip=True))
    for c in miner.mine(corpus):
        # Every surviving candidate must be exactly what the base tokenizer
        # produces for its own surface string.
        assert tuple(tok.encode(c.surface)) == c.ids


def test_candidates_never_span_a_newline_when_forbidden(base_tokenizer):
    corpus = Corpus("ja", ["子どもたちが\n公園で遊んでいます。"] * 8)
    miner = MergeCandidateMiner(base_tokenizer, MinerConfig(min_count=3, min_doc_count=2, forbid_crossing=True))
    assert all("\n" not in c.surface for c in miner.mine(corpus))


def test_raw_saving_is_count_times_length_minus_one():
    c = MergeCandidate(ids=(1, 2, 3), surface="abc", count=7)
    assert c.length == 3
    assert c.raw_saving == 14
    with pytest.raises(ValueError):
        MergeCandidate(ids=(1,), surface="a", count=1)


def test_dedupe_nested_drops_dominated_candidates():
    short = MergeCandidate(ids=(1, 2), surface="ab", count=5)
    long = MergeCandidate(ids=(1, 2, 3), surface="abc", count=5)
    other = MergeCandidate(ids=(7, 8), surface="gh", count=5)
    kept = {c.key for c in dedupe_nested([short, long, other])}
    assert long.key in kept and other.key in kept
    assert short.key not in kept, "a short candidate fully covered by a longer, equally frequent one is dead weight"


def test_mining_report_headroom_is_labelled_as_an_upper_bound(base_tokenizer, corpus):
    miner = MergeCandidateMiner(base_tokenizer, MinerConfig(min_count=3, min_doc_count=2, max_length=6))
    cands = miner.mine(corpus.head(200))
    rep = mining_report(miner, corpus.head(200), cands)
    assert rep.n_candidates == len(cands)
    assert rep.n_base_tokens > 0
    assert 0.0 <= rep.headroom <= 1.0
    assert "headroom_upper_bound" in rep.to_dict()


def test_auto_min_count_scales_with_corpus_size():
    cfg = MinerConfig(min_count=3)
    assert cfg.auto_min_count(100) == 3
    assert cfg.auto_min_count(40_000) == 100


def test_mining_is_deterministic(base_tokenizer, corpus):
    miner = MergeCandidateMiner(base_tokenizer, MinerConfig(min_count=3, min_doc_count=2, max_length=5))
    a = [c.key for c in miner.mine(corpus.head(150))]
    b = [c.key for c in miner.mine(corpus.head(150))]
    assert a == b
