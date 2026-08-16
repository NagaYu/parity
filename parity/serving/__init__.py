"""Serving one model instance under several tokenizer views.

Public surface::

    from parity import serving
    router = serving.load("Qwen/Qwen2.5-0.5B-Instruct", packs=["./packs/ja"])
    router.encode("こんにちは、世界", view="ja")     # fewer tokens
    router.encode("Hello, world", view="base")      # unchanged

Integration adapters for vLLM and TGI live in the top-level ``serving/``
directory of the repository; they wrap the same
:class:`~parity.serving.multi_tokenizer.MultiTokenizerRouter` primitives so
there is one implementation of view isolation, not three.
"""

from parity.serving.multi_tokenizer import (
    MultiTokenizerRouter,
    Request,
    Response,
    ThroughputReport,
)
from parity.serving.prefix_cache import CacheStats, PrefixCache

__all__ = [
    "MultiTokenizerRouter",
    "Request",
    "Response",
    "ThroughputReport",
    "PrefixCache",
    "CacheStats",
    "load",
]


def load(model_id: str, packs=(), device: str = "cpu", dtype: str = "float32") -> MultiTokenizerRouter:
    """Load a model and attach vocabulary packs — the three-line adoption path.

    Claim: infrastructure, reduction — if adopting a pack is harder than this,
    the reduction never reaches the people it is for.
    """
    return MultiTokenizerRouter.from_pretrained(model_id, packs=packs, device=device, dtype=dtype)
