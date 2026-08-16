"""Prefix KV cache that is safe to share across tokenizer views.

The interesting claim
---------------------
Serving several tokenizers from one model is usually assumed to fragment the
prefix cache: two requests with the same text tokenize differently, so their
KV entries cannot be shared, and the cache hit rate collapses.

Under Parity's id-space design that assumption is **false in the direction that
matters**, and the reason is worth stating precisely:

* A cache entry is keyed by a *token-id prefix*, and the KV tensors are a
  function of the id prefix and the weights alone.
* All views share one id space and one set of weights.  Id 3 means the same
  thing in every view; a pack id simply never appears in views that exclude it.
* Therefore an entry produced under view A is **numerically valid** for any
  request under view B whose id prefix matches — including the base view.

So the cache is never *corrupted* by mixing views; the only cost of multiple
views is that identical *text* under different views yields different id
sequences and therefore different keys.  That is a hit-rate question, not a
correctness question, and this module measures it:
:attr:`CacheStats.cross_view_hits` counts reuse across view boundaries, which
would be impossible if the invariant did not hold.

``tests/test_serving.py`` checks the numerical part directly: a prefix filled
under one view and reused under another produces logits identical to a cold
full forward.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("parity.serving.cache")


@dataclass
class CacheStats:
    """Counters that make the cache's behaviour auditable.

    Claim: low-cost — benchmark metric (6) is computed from these.
    """

    lookups: int = 0
    hits: int = 0
    cross_view_hits: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Share of lookups that reused at least one cached token.

        Claim: low-cost.
        """
        return self.hits / max(1, self.lookups)

    @property
    def token_reuse_rate(self) -> float:
        """Share of prompt tokens served from cache rather than recomputed.

        The number that actually maps to saved FLOPs.

        Claim: low-cost.
        """
        total = self.tokens_reused + self.tokens_computed
        return self.tokens_reused / max(1, total)

    def to_dict(self) -> Dict[str, float]:
        """Serialise.

        Claim: infrastructure.
        """
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "cross_view_hits": self.cross_view_hits,
            "tokens_reused": self.tokens_reused,
            "tokens_computed": self.tokens_computed,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "token_reuse_rate": self.token_reuse_rate,
        }


@dataclass
class CacheEntry:
    """One cached prefix.

    ``view`` records which view *produced* it — for statistics only.  It is
    deliberately not part of the key: see this module's docstring.

    Claim: low-cost, non-regression.
    """

    ids: Tuple[int, ...]
    kv: Any
    view: str = "base"
    hits: int = 0

    @property
    def length(self) -> int:
        """Number of tokens in the cached prefix.

        Claim: infrastructure.
        """
        return len(self.ids)


class _Node:
    __slots__ = ("children", "entry")

    def __init__(self) -> None:
        self.children: Dict[int, "_Node"] = {}
        self.entry: Optional[CacheEntry] = None


def clone_kv(kv: Any) -> Any:
    """Deep-copy a KV cache so a reader cannot corrupt the cached prefix.

    Handles the two shapes Parity encounters: a list of ``(k, v)`` tensor pairs
    (:class:`~parity.tiny.TinyCausalLM`, and legacy ``transformers``) and a
    ``transformers`` ``Cache`` object, which is deep-copied.

    Claim: infrastructure — sharing a mutable cache between requests is the
    classic way a serving layer produces wrong output that looks plausible.
    """
    if kv is None:
        return None
    if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], (list, tuple)):
        return [(k.clone(), v.clone()) for k, v in kv]
    crop = getattr(kv, "crop", None)
    if crop is not None:  # transformers Cache
        return copy.deepcopy(kv)
    return copy.deepcopy(kv)


def crop_kv(kv: Any, length: int) -> Any:
    """Truncate a KV cache to its first ``length`` positions.

    Claim: infrastructure — required when a stored prefix is longer than the
    matched portion of a new request.
    """
    if kv is None:
        return None
    if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], (list, tuple)):
        return [(k[:, :, :length], v[:, :, :length]) for k, v in kv]
    crop = getattr(kv, "crop", None)
    if crop is not None:
        kv = copy.deepcopy(kv)
        kv.crop(length)
        return kv
    return kv


def kv_length(kv: Any) -> int:
    """Number of cached positions in a KV cache.

    Claim: infrastructure.
    """
    if kv is None:
        return 0
    if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], (list, tuple)):
        return int(kv[0][0].shape[2])
    get_len = getattr(kv, "get_seq_length", None)
    return int(get_len()) if get_len else 0


class PrefixCache:
    """Trie-keyed longest-prefix KV cache, shared across views.

    Claim: low-cost, non-regression — the component that makes multi-tokenizer
    serving cheap without making it wrong.
    """

    def __init__(self, max_entries: int = 64, min_prefix: int = 4):
        self.max_entries = max_entries
        self.min_prefix = min_prefix
        self._root = _Node()
        self._entries: List[CacheEntry] = []
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, ids: Sequence[int], view: str = "base") -> Tuple[Optional[CacheEntry], int]:
        """Longest cached prefix of ``ids``; returns ``(entry, matched_length)``.

        The key is the id prefix only.  ``view`` is used solely to record
        whether a hit crossed a view boundary.

        Claim: low-cost.
        """
        self.stats.lookups += 1
        node = self._root
        best: Optional[CacheEntry] = None
        best_len = 0
        for i, tid in enumerate(ids):
            child = node.children.get(int(tid))
            if child is None:
                break
            node = child
            if node.entry is not None:
                best, best_len = node.entry, i + 1
        if best is not None:
            best.hits += 1
            self.stats.hits += 1
            self.stats.tokens_reused += best_len
            if best.view != view:
                self.stats.cross_view_hits += 1
        self.stats.tokens_computed += max(0, len(ids) - best_len)
        return best, best_len

    def insert(self, ids: Sequence[int], kv: Any, view: str = "base") -> Optional[CacheEntry]:
        """Store the KV state for ``ids``; returns the entry, or ``None`` if skipped.

        Claim: low-cost.
        """
        ids = tuple(int(i) for i in ids)
        if len(ids) < self.min_prefix:
            return None
        node = self._root
        for tid in ids:
            node = node.children.setdefault(tid, _Node())
        if node.entry is not None:
            return node.entry
        entry = CacheEntry(ids=ids, kv=kv, view=view)
        node.entry = entry
        self._entries.append(entry)
        self._evict_if_needed()
        return entry

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            # Least-used first; ties broken by insertion order.
            victim = min(range(len(self._entries)), key=lambda i: (self._entries[i].hits, i))
            entry = self._entries.pop(victim)
            self._unlink(entry.ids)
            self.stats.evictions += 1

    def _unlink(self, ids: Tuple[int, ...]) -> None:
        node = self._root
        path: List[Tuple[_Node, int]] = []
        for tid in ids:
            child = node.children.get(tid)
            if child is None:
                return
            path.append((node, tid))
            node = child
        node.entry = None
        for parent, tid in reversed(path):
            child = parent.children.get(tid)
            if child is not None and not child.children and child.entry is None:
                del parent.children[tid]
            else:
                break

    def clear(self) -> None:
        """Drop every entry, keeping the statistics.

        Claim: infrastructure.
        """
        self._root = _Node()
        self._entries.clear()
