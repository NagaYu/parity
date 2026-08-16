"""vLLM adapter: serve several tokenizer views from one vLLM engine.

Integration has exactly three seams, and each is a small function here so that
it can be re-pointed when vLLM's internals move:

``patch_model``
    Append the pack rows to the running engine's embedding matrix, asserting
    that no pre-existing row changed.

``ParityTokenizerRegistry``
    Resolve a request's ``view`` to the right tokenizer *before* the scheduler
    sees the prompt, so the engine only ever handles token ids.

``ViewLogitsProcessor``
    A per-request additive mask that stops a view from emitting another pack's
    tokens.

vLLM is imported lazily and never at module import time, so this file is safe to
import (and test) in an environment without it.  ``--check`` reports what it
found without loading a model.

Nothing here re-implements view semantics: it all delegates to
:class:`~parity.serving.multi_tokenizer.MultiTokenizerRouter`, because a second
implementation of "which ids may this request see" is a second thing that can be
wrong.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("parity.serving.vllm")


def vllm_available() -> bool:
    """Whether ``vllm`` can be imported here.

    Claim: infrastructure.
    """
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


class ViewLogitsProcessor:
    """Per-request additive logit mask for one tokenizer view.

    vLLM calls a logits processor as ``fn(token_ids, logits) -> logits``.  The
    mask is precomputed once per view and shared by reference across every
    request using it, so the per-step cost is one vectorised add.

    Claim: non-regression — the mechanism that keeps a base-view (English)
    request numerically identical to the original model, and stops one
    language's pack from leaking into another's output.
    """

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, token_ids: Sequence[int], logits):
        """Apply the mask in place-ish and return the logits.

        Claim: non-regression.
        """
        return logits + self.mask.to(logits.device, logits.dtype)


class ParityTokenizerRegistry:
    """Resolves ``view -> tokenizer`` and encodes prompts before scheduling.

    Claim: reduction, non-regression — the point at which a request actually
    stops paying the token premium, and the point at which view isolation is
    enforced.
    """

    def __init__(self, router):
        self.router = router

    def views(self) -> List[str]:
        """Available view names.

        Claim: infrastructure.
        """
        return self.router.views()

    def encode(self, prompt: str, view: str = "base") -> List[int]:
        """Encode a prompt under a view.

        Claim: reduction.
        """
        return self.router.encode(prompt, view)

    def decode(self, ids: Sequence[int]) -> str:
        """Decode output ids; view-independent by construction.

        Claim: non-regression.
        """
        return self.router.decode(ids)

    def logits_processor(self, view: str) -> ViewLogitsProcessor:
        """Build the per-request logits processor for a view.

        Claim: non-regression.
        """
        return ViewLogitsProcessor(self.router.logit_mask(view))

    def sampling_params(self, view: str, **kw):
        """``vllm.SamplingParams`` pre-loaded with this view's mask.

        Claim: non-regression, infrastructure.
        """
        from vllm import SamplingParams

        procs = list(kw.pop("logits_processors", []) or [])
        procs.append(self.logits_processor(view))
        return SamplingParams(logits_processors=procs, **kw)


def patch_model(llm, packs: Sequence[Any], tokenizer=None):
    """Append pack rows to a live vLLM engine and return a router over it.

    The append-only assertion in
    :meth:`~parity.adapters.TorchLMAdapter.append_rows` runs here too, so a pack
    that would disturb the base model fails at load rather than in production.

    Claim: non-regression, reduction — the one function that turns a stock vLLM
    deployment into a multi-view one.
    """
    from parity.adapters import TorchLMAdapter
    from parity.serving import MultiTokenizerRouter
    from parity.tokenization import AugmentedTokenizer

    model = _unwrap_model(llm)
    adapter = TorchLMAdapter(model, name=getattr(llm, "model_id", "vllm"))
    tokenizer = tokenizer or _unwrap_tokenizer(llm)
    aug = AugmentedTokenizer(tokenizer)
    for pack in packs:
        aug.attach(pack)
        adapter.append_rows(pack.input_embeddings, pack.output_embeddings)
    aug.check_invariants()
    log.info("patched vLLM engine: %d views, %d added rows", len(aug.packs()) + 1, aug.n_added)
    return MultiTokenizerRouter(adapter, aug)


def _unwrap_model(llm):
    """Reach the ``nn.Module`` inside a vLLM ``LLM`` handle.

    vLLM has moved this path more than once; each candidate is tried in turn and
    the failure message names them all rather than raising an opaque
    ``AttributeError``.

    Claim: infrastructure.
    """
    candidates = [
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.model_runner.model,
        lambda: llm.model,
    ]
    for get in candidates:
        try:
            model = get()
            if model is not None:
                return model
        except Exception:
            continue
    raise RuntimeError(
        "could not locate the nn.Module inside this vLLM engine; vLLM's internals have moved. "
        "Point _unwrap_model at the current path — it is the only thing that needs changing."
    )


def _unwrap_tokenizer(llm):
    """Reach the HF tokenizer inside a vLLM ``LLM`` handle.

    Claim: infrastructure.
    """
    from parity.tokenization import HFTokenizer

    for get in (lambda: llm.get_tokenizer(), lambda: llm.llm_engine.tokenizer.tokenizer):
        try:
            tok = get()
            if tok is not None:
                return HFTokenizer(tok)
        except Exception:
            continue
    raise RuntimeError("could not locate the tokenizer inside this vLLM engine")


def generate(llm, registry: ParityTokenizerRegistry, prompts: Sequence[Dict[str, Any]], **sampling):
    """Run a mixed-view batch through vLLM.

    ``prompts`` is a list of ``{"prompt": str, "view": str}``.  Every request is
    encoded under its own view and submitted as token ids, so the engine batches
    them together exactly as it would a single-view batch.

    Claim: low-cost, reduction — the demonstration that mixed-view serving is
    ordinary batched serving.
    """
    from vllm import TokensPrompt

    requests, params = [], []
    for p in prompts:
        view = p.get("view", "base")
        requests.append(TokensPrompt(prompt_token_ids=registry.encode(p["prompt"], view)))
        params.append(registry.sampling_params(view, **sampling))
    outputs = llm.generate(requests, params)
    return [
        {
            "view": p.get("view", "base"),
            "text": registry.decode(o.outputs[0].token_ids),
            "prompt_tokens": len(o.prompt_token_ids or []),
            "generated_tokens": len(o.outputs[0].token_ids),
        }
        for p, o in zip(prompts, outputs)
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: check the environment, or start a patched engine and run a demo batch.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Serve Parity packs from vLLM")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--packs", nargs="*", default=[])
    ap.add_argument("--check", action="store_true", help="report the environment and exit")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--view", default="base")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.check or not vllm_available():
        print(f"vllm importable: {vllm_available()}")
        print(f"packs requested: {args.packs}")
        if not vllm_available():
            print("install vllm to use this plugin: pip install vllm")
            print("the engine-agnostic router in parity.serving works without it")
        return 0

    from vllm import LLM

    from parity.pack import load_pack

    llm = LLM(model=args.model)
    router = patch_model(llm, [load_pack(p) for p in args.packs])
    registry = ParityTokenizerRegistry(router)
    print("views:", registry.views())
    if args.prompt:
        for r in generate(llm, registry, [{"prompt": args.prompt, "view": args.view}], max_tokens=64):
            print(r)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
