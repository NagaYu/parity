"""Tokenizer adapters and the augmented (multi-view) tokenizer.

The central design decision of Parity lives here: **new tokens are defined in
token-id space, not string space.**  A Parity token is a contiguous run of base
token ids, and encoding is

    base_encode(text)  ->  leftmost-longest merge pass  ->  augmented ids

with the exact inverse available as :meth:`AugmentedTokenizer.to_base_ids`.

Three properties follow, and all three are load-bearing:

1. **Lossless round-trip for any base tokenizer.**  BPE, Unigram, WordPiece and
   byte-level tokenizers all differ in how they pre-tokenize; none of that
   matters if we only ever merge ids the base tokenizer already emitted.
   ``decode(encode(x)) == base_decode(base_encode(x))`` by construction.

2. **Exact English non-regression.**  Ids are append-only: base ids keep their
   indices, pack ids start at ``base_vocab_size``.  A request that selects the
   base view provably cannot see or emit a pack token, so its logits are the
   original model's logits bit-for-bit.

3. **Prefix-cache safety across views.**  Because all views share one id space
   and one set of weights, a KV entry keyed by an id prefix is valid for every
   view that could have produced it — views differ in what they *emit*, never
   in what an id *means*.  See :mod:`parity.serving.prefix_cache`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

log = logging.getLogger("parity.tokenization")

#: Fixed probe strings used to fingerprint tokenizer *behaviour* (see
#: :func:`parity.types.tokenizer_fingerprint`).  They deliberately span scripts
#: so that two tokenizers that agree on English but differ elsewhere do not
#: collide.
FINGERPRINT_PROBES = (
    "The quick brown fox jumps over the lazy dog.",
    "子どもたちが公園で遊んでいます。",
    "बच्चे पार्क में खेल रहे हैं।",
    "الأطفال يلعبون في الحديقة.",
    "เด็ก ๆ กำลังเล่นอยู่ในสวนสาธารณะ",
    "Watoto wanacheza bustanini.",
)


@runtime_checkable
class BaseTokenizer(Protocol):
    """Minimal interface Parity needs from a tokenizer.

    Claim: infrastructure — deliberately three methods, so a language community
    can plug in a tokenizer we have never heard of.
    """

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class HFTokenizer:
    """Adapter around a ``transformers`` tokenizer.

    Special tokens are disabled on encode: Parity measures and mines the cost of
    *content*, and a BOS token added to every sentence would inflate short-
    sentence fertility uniformly and differently per model.

    Claim: infrastructure.
    """

    def __init__(self, tok, name: str = ""):
        self._tok = tok
        self.name = name or getattr(tok, "name_or_path", "hf-tokenizer")

    @classmethod
    def from_pretrained(cls, model_id: str, **kw) -> "HFTokenizer":
        """Load a tokenizer from the Hub or a local path.

        Claim: infrastructure.
        """
        from transformers import AutoTokenizer  # local import: keeps torch-free paths fast

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kw)
        return cls(tok, name=model_id)

    @property
    def vocab_size(self) -> int:
        """Size of the id space, including any already-added special tokens.

        ``len(tok)`` rather than ``tok.vocab_size`` because the latter excludes
        added tokens, and appending pack ids on top of an id that is already in
        use would corrupt the model silently.

        Claim: non-regression — off-by-one here is a wrong-token bug.
        """
        return len(self._tok)

    def encode(self, text: str) -> List[int]:
        """Encode without special tokens.

        Claim: reduction.
        """
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        """Decode, keeping special tokens out of the surface string.

        Claim: infrastructure.
        """
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """Sub-word strings, for human inspection in the Space and the miner.

        Claim: infrastructure.
        """
        try:
            return self._tok.convert_ids_to_tokens(list(ids))
        except Exception:  # pragma: no cover
            return [self.decode([i]) for i in ids]

    @property
    def hf(self):
        """The wrapped ``transformers`` tokenizer, for export paths.

        Claim: infrastructure.
        """
        return self._tok


class ByteTokenizer:
    """A UTF-8 byte tokenizer: the honest worst case, and an offline stand-in.

    Fertility under this tokenizer is exactly "bytes per character", which for
    Latin text is ~1 and for Devanagari/Thai/Ethiopic is 3.  It is used in tests
    as a base tokenizer that needs no download and still exhibits the real
    cross-script asymmetry Parity exists to fix.

    Claim: infrastructure.
    """

    def __init__(self, n_special: int = 4):
        self.n_special = n_special
        self.name = "byte-utf8"

    @property
    def vocab_size(self) -> int:
        """256 byte values plus a few reserved special ids.

        Claim: infrastructure.
        """
        return 256 + self.n_special

    def encode(self, text: str) -> List[int]:
        """UTF-8 bytes, offset past the special-id block.

        Claim: reduction.
        """
        return [b + self.n_special for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of :meth:`encode`; unknown/special ids are dropped.

        Claim: infrastructure.
        """
        data = bytes(i - self.n_special for i in ids if self.n_special <= i < 256 + self.n_special)
        return data.decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """Per-byte display strings.

        Claim: infrastructure.
        """
        return [f"<{i - self.n_special:02x}>" for i in ids]


# ---------------------------------------------------------------------------
# Merge trie
# ---------------------------------------------------------------------------


class MergeTrie:
    """Leftmost-longest matcher over base-token-id sequences.

    A plain dict trie: ``children`` maps an id to a child node, ``token`` marks
    a node as terminal with the augmented id it produces.  Matching is a single
    left-to-right pass that, at each position, walks as deep as it can and emits
    the deepest terminal it saw.

    Leftmost-longest (rather than, say, frequency-ordered) matching is chosen
    because it is *deterministic and order-independent*: the same pack always
    produces the same ids regardless of the order tokens were added, which is
    what lets a prefix cache key be a pure function of the id prefix.

    Claim: reduction, infrastructure — the mechanism that turns adopted
    candidates into actual token savings.
    """

    __slots__ = ("children", "token", "depth", "_n")

    def __init__(self) -> None:
        self.children: Dict[int, "MergeTrie"] = {}
        self.token: Optional[int] = None
        self.depth: int = 0
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, ids: Sequence[int], new_id: int) -> None:
        """Insert ``ids -> new_id``.

        Claim: reduction.
        """
        node = self
        for depth, i in enumerate(ids, start=1):
            child = node.children.get(i)
            if child is None:
                child = MergeTrie()
                child.depth = depth
                node.children[i] = child
            node = child
        if node.token is None:
            self._n += 1
        node.token = new_id

    def merge(self, ids: Sequence[int]) -> List[int]:
        """Apply one leftmost-longest merge pass.

        Runs in O(len(ids) * max_pattern_len) with no allocation per position
        beyond the output list.

        Claim: reduction — this is where the token count actually drops.
        """
        out: List[int] = []
        i, n = 0, len(ids)
        while i < n:
            node = self
            best_tok: Optional[int] = None
            best_len = 0
            j = i
            while j < n:
                child = node.children.get(ids[j])
                if child is None:
                    break
                node = child
                j += 1
                if node.token is not None:
                    best_tok, best_len = node.token, j - i
            if best_tok is not None:
                out.append(best_tok)
                i += best_len
            else:
                out.append(ids[i])
                i += 1
        return out

    def occurrences(self, ids: Sequence[int]) -> List[Tuple[int, int, int]]:
        """All leftmost-longest matches as ``(start, length, new_id)``.

        Used by the selection stage to compute exact retokenised lengths without
        materialising the merged sequence.

        Claim: reduction.
        """
        spans: List[Tuple[int, int, int]] = []
        i, n = 0, len(ids)
        while i < n:
            node = self
            best_tok, best_len = None, 0
            j = i
            while j < n:
                child = node.children.get(ids[j])
                if child is None:
                    break
                node = child
                j += 1
                if node.token is not None:
                    best_tok, best_len = node.token, j - i
            if best_tok is not None:
                spans.append((i, best_len, best_tok))
                i += best_len
            else:
                i += 1
        return spans


# ---------------------------------------------------------------------------
# Views and the augmented tokenizer
# ---------------------------------------------------------------------------


@dataclass
class TokenizerView:
    """A named subset of packs: what one request is allowed to use.

    ``BASE`` is the distinguished view containing no packs.  Serving one model
    with several views is the whole point of :mod:`parity.serving`.

    Claim: non-regression — the view is the object that makes "English is
    untouched" a checkable statement rather than a hope.
    """

    name: str
    pack_langs: Tuple[str, ...] = ()
    allowed_ids: Optional[frozenset] = None  # None == base ids only
    #: When true the view *reads* pack tokens but may never *emit* them.
    #:
    #: This is the right mode for a model with **tied** embeddings, where the
    #: input row is also the output row and the two objectives compete: an
    #: embedding that reproduces the expansion's internal state is not generally
    #: the one that gives the token the right emission probability. Refusing to
    #: emit removes the second objective entirely, so emission drift is zero by
    #: construction rather than bounded by measurement.
    #:
    #: The saving is essentially unaffected. Prompt tokens are where the cost is,
    #: and generation continues to use base tokens — which decode to exactly the
    #: same strings.
    input_only: bool = False

    @property
    def is_base(self) -> bool:
        """True for the view that uses no pack tokens at all.

        Claim: non-regression.
        """
        return not self.pack_langs


BASE_VIEW = TokenizerView("base")


class AugmentedTokenizer:
    """Base tokenizer + zero or more :class:`~parity.types.VocabPack` s.

    Id layout::

        [0, base_vocab_size)                    base tokenizer ids, untouched
        [base_vocab_size, base_vocab_size + N)  pack tokens, in attach order

    Attaching a pack never renumbers anything that already existed.  This is the
    invariant behind exact English non-regression and cross-view prefix-cache
    reuse; :meth:`attach` enforces it and :meth:`check_invariants` re-checks it.

    Claim: reduction, non-regression — encodes fewer tokens for pack languages
    while leaving the base id space bit-identical.
    """

    def __init__(self, base: BaseTokenizer, base_vocab_size: Optional[int] = None):
        self.base = base
        self.base_vocab_size = int(base_vocab_size if base_vocab_size is not None else base.vocab_size)
        self._next_id = self.base_vocab_size
        self._expansions: Dict[int, Tuple[int, ...]] = {}
        self._surface: Dict[int, str] = {}
        self._lang_of: Dict[int, str] = {}
        self._packs: Dict[str, List[int]] = {}
        self._tries: Dict[str, MergeTrie] = {}
        self._view_cache: Dict[Tuple[str, ...], MergeTrie] = {}

    # -- construction -------------------------------------------------------

    def attach(self, pack) -> List[int]:
        """Add a pack's tokens, assigning fresh contiguous ids.

        Raises if the pack was built against a different tokenizer fingerprint
        or a different base vocab size — silently mismatched packs would produce
        embeddings for the wrong sub-words, which is exactly the failure mode a
        certificate cannot catch after the fact.

        Claim: non-regression — append-only id assignment is the mechanism that
        makes the base view exactly equal to the original model.
        """
        if pack.base_vocab_size != self.base_vocab_size:
            raise ValueError(
                f"pack '{pack.lang}' was built for base_vocab_size={pack.base_vocab_size}, "
                f"tokenizer has {self.base_vocab_size}"
            )
        expected = self.fingerprint()
        if pack.base_tokenizer_fingerprint and pack.base_tokenizer_fingerprint != expected:
            raise ValueError(
                f"pack '{pack.lang}' was built against a different tokenizer "
                f"({pack.base_tokenizer_fingerprint[:12]}… != {expected[:12]}…)"
            )
        if pack.lang in self._packs:
            raise ValueError(f"pack for {pack.lang!r} already attached")

        assigned: List[int] = []
        trie = MergeTrie()
        for entry in pack.entries:
            new_id = self._next_id
            self._next_id += 1
            entry.new_id = new_id
            self._expansions[new_id] = tuple(entry.candidate.ids)
            self._surface[new_id] = entry.candidate.surface
            self._lang_of[new_id] = pack.lang
            trie.add(entry.candidate.ids, new_id)
            assigned.append(new_id)
        self._packs[pack.lang] = assigned
        self._tries[pack.lang] = trie
        self._view_cache.clear()
        log.info("attached pack %s: %d tokens, ids [%d, %d)", pack.lang, len(assigned), assigned[0] if assigned else -1, self._next_id)
        return assigned

    def add_tokens(self, expansions: Iterable[Sequence[int]], lang: str = "adhoc") -> List[int]:
        """Attach raw id-sequences without a full pack — used by tests and the miner preview.

        Claim: infrastructure.
        """
        assigned: List[int] = []
        trie = self._tries.setdefault(lang, MergeTrie())
        bucket = self._packs.setdefault(lang, [])
        for ids in expansions:
            ids = tuple(int(i) for i in ids)
            new_id = self._next_id
            self._next_id += 1
            self._expansions[new_id] = ids
            self._surface[new_id] = self.base.decode(ids)
            self._lang_of[new_id] = lang
            trie.add(ids, new_id)
            assigned.append(new_id)
            bucket.append(new_id)
        self._view_cache.clear()
        return assigned

    # -- views --------------------------------------------------------------

    @property
    def total_vocab_size(self) -> int:
        """Base vocab plus every attached pack token.

        Claim: infrastructure.
        """
        return self._next_id

    @property
    def n_added(self) -> int:
        """Number of Parity tokens attached.

        Claim: low-cost — this times ``d_model`` (times 2 if untied) is the
        entire parameter cost of the method.
        """
        return self._next_id - self.base_vocab_size

    def packs(self) -> List[str]:
        """Attached pack languages.

        Claim: infrastructure.
        """
        return sorted(self._packs)

    def view(self, *langs: str, input_only: bool = False) -> TokenizerView:
        """Build a view enabling the named packs (none == the base view).

        ``input_only=True`` lets the view read pack tokens but never emit them;
        see :attr:`TokenizerView.input_only` for why that is the right default
        on tied-embedding models.

        Claim: non-regression.
        """
        langs = tuple(sorted(l for l in langs if l))
        for l in langs:
            if l not in self._packs:
                raise KeyError(f"no pack attached for {l!r}; attached: {self.packs()}")
        if not langs:
            return BASE_VIEW
        allowed = frozenset(range(self.base_vocab_size)) | frozenset(i for l in langs for i in self._packs[l])
        name = "+".join(langs) + (":in" if input_only else "")
        return TokenizerView(name, langs, allowed, input_only=input_only)

    def _trie_for(self, view: TokenizerView) -> Optional[MergeTrie]:
        if view.is_base:
            return None
        key = tuple(view.pack_langs)
        cached = self._view_cache.get(key)
        if cached is None:
            cached = MergeTrie()
            for lang in key:
                for new_id in self._packs[lang]:
                    cached.add(self._expansions[new_id], new_id)
            self._view_cache[key] = cached
        return cached

    # -- encode / decode ----------------------------------------------------

    def encode(self, text: str, view: TokenizerView = BASE_VIEW) -> List[int]:
        """Encode under ``view``: base encode, then one merge pass.

        Claim: reduction — the observable effect of a pack is that this returns
        a shorter list than the base tokenizer for the pack's language, and an
        identical list for every other input.
        """
        ids = self.base.encode(text)
        trie = self._trie_for(view)
        return trie.merge(ids) if trie is not None else ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode augmented ids by expanding pack tokens, then base-decoding.

        View-independent on purpose: an id means the same thing everywhere, so
        a detokenizer never needs to know which view produced the ids.

        Claim: non-regression — round-trip fidelity is checked in
        ``tests/test_tokenization.py``.
        """
        return self.base.decode(self.to_base_ids(ids))

    def to_base_ids(self, ids: Sequence[int]) -> List[int]:
        """Expand every pack token back to its base id sequence.

        Claim: bound — the certificate is defined as a comparison against
        *this* sequence, so the expansion must be exact.
        """
        out: List[int] = []
        for i in ids:
            exp = self._expansions.get(int(i))
            if exp is None:
                out.append(int(i))
            else:
                out.extend(exp)
        return out

    def expansion(self, new_id: int) -> Tuple[int, ...]:
        """Base id sequence behind a pack token.

        Claim: infrastructure.
        """
        return self._expansions[int(new_id)]

    def surface(self, new_id: int) -> str:
        """Human-readable string for a pack token.

        Claim: infrastructure.
        """
        return self._surface.get(int(new_id), "")

    def lang_of(self, new_id: int) -> str:
        """Which pack a token came from.

        Claim: infrastructure.
        """
        return self._lang_of.get(int(new_id), "")

    def count(self, text: str, view: TokenizerView = BASE_VIEW) -> int:
        """Token count under ``view`` — the quantity users are billed for.

        Claim: reduction.
        """
        return len(self.encode(text, view))

    # -- integrity ----------------------------------------------------------

    def fingerprint(self) -> str:
        """Behavioural fingerprint of the *base* tokenizer.

        Claim: infrastructure — binds a pack to the tokenizer it was mined on.
        """
        from parity.types import tokenizer_fingerprint

        return tokenizer_fingerprint(self.base_vocab_size, [self.base.encode(p) for p in FINGERPRINT_PROBES])

    def check_invariants(self, probes: Sequence[str] = FINGERPRINT_PROBES) -> None:
        """Assert the three properties this module's docstring promises.

        Raises ``AssertionError`` with a specific message on violation.  Called
        by the CLI after building a pack and by ``tests/test_tokenization.py``.

        Claim: non-regression, reduction — turns the design's guarantees into
        an executable check rather than prose.
        """
        for text in probes:
            base_ids = self.base.encode(text)
            # (1) base view is byte-identical to the base tokenizer
            assert self.encode(text, BASE_VIEW) == base_ids, "base view diverged from the base tokenizer"
            for lang in self.packs():
                v = self.view(lang)
                aug = self.encode(text, v)
                # (2) lossless expansion
                assert self.to_base_ids(aug) == base_ids, f"expansion of view {v.name} is not the base sequence"
                # (3) never longer than the base encoding
                assert len(aug) <= len(base_ids), f"view {v.name} made {text[:20]!r} longer"
                # (4) view isolation: no id outside the view's allowance
                if v.allowed_ids is not None:
                    assert all(i in v.allowed_ids for i in aug), f"view {v.name} emitted a foreign id"
        # (5) append-only id space
        assert min(self._expansions, default=self.base_vocab_size) >= self.base_vocab_size, (
            "a pack token was assigned an id inside the base vocabulary"
        )


@dataclass
class TokenizationDiff:
    """Side-by-side base vs augmented tokenization of one string.

    Powers the Space's "here is what you were paying for" panel.

    Claim: reduction.
    """

    text: str
    base_ids: List[int] = field(default_factory=list)
    aug_ids: List[int] = field(default_factory=list)
    base_pieces: List[str] = field(default_factory=list)
    aug_pieces: List[str] = field(default_factory=list)

    @property
    def reduction(self) -> float:
        """Fraction of tokens removed, in ``[0, 1)``.

        Claim: reduction — the headline number, per string.
        """
        if not self.base_ids:
            return 0.0
        return 1.0 - len(self.aug_ids) / len(self.base_ids)


def diff_tokenization(tok: AugmentedTokenizer, text: str, view: TokenizerView) -> TokenizationDiff:
    """Tokenize ``text`` both ways and return the aligned pieces.

    Claim: reduction — this is the demo that makes the saving visible rather
    than asserted.
    """
    base_ids = tok.encode(text, BASE_VIEW)
    aug_ids = tok.encode(text, view)
    conv = getattr(tok.base, "convert_ids_to_tokens", None)
    base_pieces = conv(base_ids) if conv else [tok.base.decode([i]) for i in base_ids]
    aug_pieces = []
    for i in aug_ids:
        if i >= tok.base_vocab_size:
            aug_pieces.append(tok.surface(i))
        else:
            aug_pieces.append(conv([i])[0] if conv else tok.base.decode([i]))
    return TokenizationDiff(text, list(base_ids), list(aug_ids), list(base_pieces), aug_pieces)
