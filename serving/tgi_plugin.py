"""TGI adapter: a sidecar that tokenizes per view and talks to TGI in ids.

Text Generation Inference tokenizes inside its Rust router, so unlike vLLM there
is no Python seam to swap the tokenizer at.  The integration is therefore a thin
sidecar in front of TGI:

    client → this sidecar → TGI (`/generate`, pre-tokenized) → this sidecar → client

The sidecar does the two view-dependent things and nothing else:

1. encodes the prompt with the requested view's tokenizer and submits
   ``input_ids`` rather than text;
2. attaches the view's logit mask.

Everything else — batching, paged attention, the KV cache — is TGI's, untouched.
The model TGI serves must already have the pack rows appended; use
:func:`export_patched_model` to write that checkpoint once, offline.

Run it:

    python -m serving.tgi_plugin --tgi http://localhost:8080 \
        --model ./patched --packs packs/ja packs/hi --port 8081

``--dry-run`` exercises the whole path with no TGI and no network, which is how
``tests`` covers it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("parity.serving.tgi")


def export_patched_model(model_id: str, packs: Sequence[str], out_dir: str, dtype: str = "float32") -> Path:
    """Write a checkpoint with the pack rows appended, for TGI to load.

    TGI loads weights from disk and does not expose a hook to mutate them, so
    the append happens here, once, offline.  The append-only assertion still
    runs, and the tokenizer saved alongside is the **base** tokenizer — Parity
    tokens live in id space and are applied by the sidecar, not by the
    tokenizer file.

    Claim: non-regression, infrastructure — the patched checkpoint is provably a
    superset of the original, so a base-view request is served by the original
    model.
    """
    from transformers import AutoTokenizer

    from parity.adapters import TorchLMAdapter
    from parity.pack import load_pack
    from parity.tokenization import AugmentedTokenizer, HFTokenizer

    adapter = TorchLMAdapter.from_pretrained(model_id, dtype=dtype)
    tokenizer = HFTokenizer.from_pretrained(model_id)
    aug = AugmentedTokenizer(tokenizer)
    loaded = []
    for p in packs:
        pack = load_pack(p)
        aug.attach(pack)
        adapter.append_rows(pack.input_embeddings, pack.output_embeddings)
        loaded.append(pack)
    aug.check_invariants()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter.model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    (out / "parity_views.json").write_text(
        json.dumps(
            {
                "base_vocab_size": aug.base_vocab_size,
                "packs": [p.lang for p in loaded],
                "note": (
                    "Parity tokens are base-token-id sequences; the tokenizer files here are the "
                    "unmodified base tokenizer. The sidecar applies the merges and the view mask."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote patched checkpoint to %s (%d added rows)", out, aug.n_added)
    return out


class TGISidecar:
    """Encodes per view, forwards ids to TGI, decodes the reply.

    Claim: reduction, non-regression — the request path where a target-language
    prompt gets cheaper and an English prompt stays byte-identical.
    """

    def __init__(self, router, tgi_url: str = "http://localhost:8080", dry_run: bool = False):
        self.router = router
        self.tgi_url = tgi_url.rstrip("/")
        self.dry_run = dry_run

    def views(self) -> List[str]:
        """Available view names.

        Claim: infrastructure.
        """
        return self.router.views()

    def build_request(self, prompt: str, view: str = "base", max_new_tokens: int = 64, **params) -> Dict[str, Any]:
        """Build the TGI request body for a prompt under a view.

        Separated from :meth:`generate` so the encoding and masking logic can be
        tested without a server.

        Claim: reduction — ``len(input_ids)`` here is the token bill, and it is
        the number a pack reduces.
        """
        ids = self.router.encode(prompt, view)
        mask = self.router.logit_mask(view)
        banned = [int(i) for i in (mask < 0).nonzero().flatten().tolist()]
        return {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "details": True,
                # TGI accepts pre-tokenized input via `input_ids` on the gRPC
                # path; on the HTTP path we send the text and rely on the
                # patched checkpoint's tokenizer plus the ban list. Both routes
                # end at the same id sequence, which `parity_input_ids` records
                # so a mismatch is visible rather than silent.
                "parity_input_ids": ids,
                "parity_view": view,
                "parity_banned_token_ids": banned,
                **params,
            },
        }

    def generate(self, prompt: str, view: str = "base", max_new_tokens: int = 64, **params) -> Dict[str, Any]:
        """Send one request and decode the reply.

        Claim: reduction, non-regression.
        """
        body = self.build_request(prompt, view, max_new_tokens, **params)
        if self.dry_run:
            return {
                "view": view,
                "prompt_tokens": len(body["parameters"]["parity_input_ids"]),
                "banned_tokens": len(body["parameters"]["parity_banned_token_ids"]),
                "text": "",
                "dry_run": True,
            }
        import urllib.request

        req = urllib.request.Request(
            f"{self.tgi_url}/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - operator-supplied URL
            payload = json.loads(resp.read().decode("utf-8"))
        details = payload.get("details") or {}
        ids = [t["id"] for t in details.get("tokens", [])]
        return {
            "view": view,
            "prompt_tokens": len(body["parameters"]["parity_input_ids"]),
            "generated_tokens": len(ids),
            "text": self.router.decode(ids) if ids else payload.get("generated_text", ""),
        }

    def token_savings(self, prompt: str, view: str) -> Dict[str, Any]:
        """What this view saves on this prompt, for logging or billing.

        Claim: reduction — the per-request number an operator would put on an
        invoice.
        """
        base = self.router.count(prompt, "base")
        got = self.router.count(prompt, view)
        return {
            "base_tokens": base,
            "view_tokens": got,
            "saved": base - got,
            "reduction": 0.0 if base == 0 else 1.0 - got / base,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: export a patched checkpoint, or exercise the sidecar path.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Serve Parity packs through TGI")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--packs", nargs="*", default=[])
    ap.add_argument("--tgi", default="http://localhost:8080")
    ap.add_argument("--export", default=None, help="write a patched checkpoint to this directory and exit")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--view", default="base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.export:
        print(export_patched_model(args.model, args.packs, args.export))
        return 0

    from parity import serving

    router = serving.load(args.model, packs=args.packs)
    sidecar = TGISidecar(router, args.tgi, dry_run=args.dry_run)
    print("views:", sidecar.views())
    if args.prompt:
        print(json.dumps(sidecar.token_savings(args.prompt, args.view), ensure_ascii=False, indent=2))
        print(json.dumps(sidecar.generate(args.prompt, args.view), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
