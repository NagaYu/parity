"""Tokenizer-level invariants: round-trip, append-only ids, view isolation.

These are the properties the rest of the project builds on.  If any of them
breaks, the English non-regression claim and the prefix-cache-sharing claim both
become false, so they are tested first and directly.
"""

from __future__ import annotations

import pytest

from parity.tokenization import BASE_VIEW, AugmentedTokenizer, ByteTokenizer, MergeTrie


def test_merge_trie_is_leftmost_longest():
    trie = MergeTrie()
    trie.add([1, 2], 100)
    trie.add([1, 2, 3], 101)
    trie.add([2, 3], 102)
    # The longest match starting at position 0 wins, and consumes 2,3 with it.
    assert trie.merge([1, 2, 3]) == [101]
    assert trie.merge([0, 1, 2, 9]) == [0, 100, 9]
    assert trie.merge([0, 2, 3]) == [0, 102]
    # Order of insertion must not matter.
    other = MergeTrie()
    for ids, tid in [([2, 3], 102), ([1, 2, 3], 101), ([1, 2], 100)]:
        other.add(ids, tid)
    assert other.merge([1, 2, 3]) == trie.merge([1, 2, 3])


def test_merge_trie_id_zero_is_a_valid_token():
    # `0` is falsy; a `if node.token:` bug would silently drop it.
    trie = MergeTrie()
    trie.add([7, 8], 0)
    assert trie.merge([7, 8]) == [0]


def test_byte_tokenizer_round_trips():
    tok = ByteTokenizer()
    for text in ["hello", "日本語のテキスト", "الأطفال", "เด็ก"]:
        assert tok.decode(tok.encode(text)) == text


def test_base_view_is_identical_to_the_base_tokenizer(plain_augmented, sample):
    aug = plain_augmented
    for text in sample.by_lang["en"] + sample.by_lang["ja"]:
        assert aug.encode(text, BASE_VIEW) == aug.base.encode(text)


def test_added_tokens_are_append_only_and_reduce_length(base_tokenizer, sample):
    aug = AugmentedTokenizer(base_tokenizer)
    base_v = aug.base_vocab_size
    text = sample.by_lang["ja"][0]
    ids = base_tokenizer.encode(text)
    assert len(ids) >= 4
    new_ids = aug.add_tokens([tuple(ids[:3])], lang="ja")

    assert min(new_ids) >= base_v, "a pack token was given an id inside the base vocabulary"
    view = aug.view("ja")
    merged = aug.encode(text, view)
    assert len(merged) < len(ids)
    assert aug.to_base_ids(merged) == ids, "expansion is not the exact base sequence"
    assert aug.decode(merged) == aug.base.decode(ids)
    # And the base view is untouched by the attachment.
    assert aug.encode(text, BASE_VIEW) == ids


def test_view_isolation_never_emits_a_foreign_id(base_tokenizer, sample):
    aug = AugmentedTokenizer(base_tokenizer)
    ja_ids = base_tokenizer.encode(sample.by_lang["ja"][0])
    hi_ids = base_tokenizer.encode(sample.by_lang["hi"][0])
    aug.add_tokens([tuple(ja_ids[:3])], lang="ja")
    aug.add_tokens([tuple(hi_ids[:3])], lang="hi")

    ja_view, hi_view = aug.view("ja"), aug.view("hi")
    out = aug.encode(sample.by_lang["ja"][0], ja_view)
    assert all(i in ja_view.allowed_ids for i in out)
    # The Hindi view must not be able to produce the Japanese pack token.
    out_hi = aug.encode(sample.by_lang["ja"][0], hi_view)
    assert all(i in hi_view.allowed_ids for i in out_hi)
    assert out_hi == ja_ids


def test_check_invariants_runs_on_a_real_pack(attached, sample):
    aug = attached["tokenizer"]
    aug.check_invariants()
    for text in sample.by_lang["ja"]:
        merged = aug.encode(text, aug.view("ja"))
        assert aug.to_base_ids(merged) == aug.base.encode(text)
        assert aug.decode(merged) == aug.base.decode(aug.base.encode(text))


def test_attaching_a_mismatched_pack_is_refused(built, base_tokenizer):
    import dataclasses

    aug = AugmentedTokenizer(base_tokenizer)
    pack = built["result"].pack
    bad = dataclasses.replace(pack, base_vocab_size=pack.base_vocab_size + 1)
    with pytest.raises(ValueError, match="base_vocab_size"):
        aug.attach(bad)

    bad2 = dataclasses.replace(pack, base_tokenizer_fingerprint="deadbeef" * 8)
    with pytest.raises(ValueError, match="different tokenizer"):
        aug.attach(bad2)
