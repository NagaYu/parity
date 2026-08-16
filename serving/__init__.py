"""Engine adapters: vLLM and TGI.

Both delegate view semantics to
:class:`~parity.serving.multi_tokenizer.MultiTokenizerRouter` rather than
reimplementing them, because "which token ids may this request see" is a safety
property, and a second implementation of a safety property is a second thing
that can be wrong.

Imports of ``vllm`` and network calls are lazy, so this package is importable
(and testable) without either.
"""

__all__ = ["vllm_plugin", "tgi_plugin"]
