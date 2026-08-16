"""Vocabulary pack I/O and model-card generation.

A pack is a directory::

    pack/
      manifest.json          candidates, ids, certificates, provenance
      embeddings.safetensors input_embeddings [n, d] (+ output_embeddings)
      added_tokens.json      surface strings, for tools that want them
      README.md              the model card, with the certified drift in it

Two rules are enforced on load, not documented and hoped for:

* the pack's base tokenizer fingerprint must match the tokenizer it is attached
  to (:meth:`parity.tokenization.AugmentedTokenizer.attach`);
* every entry must carry an accepted certificate.

The model card is generated, never hand-written, so the drift numbers on the
card are the drift numbers in the manifest by construction.  A card that says
"safe" while the manifest says otherwise is the single most damaging failure
mode this project could have.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from parity.types import DriftCertificate, PackEntry, VocabPack

log = logging.getLogger("parity.pack")

MANIFEST = "manifest.json"
TENSORS = "embeddings.safetensors"
ADDED_TOKENS = "added_tokens.json"
CARD = "README.md"


def save_pack(pack: VocabPack, path: str | Path) -> Path:
    """Write a pack to ``path`` (created if needed) and return the directory.

    Claim: infrastructure — the deliverable format that carries reduction,
    non-regression and bound evidence together, so none can be shipped without
    the others.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    (out / MANIFEST).write_text(json.dumps(pack.manifest(), ensure_ascii=False, indent=2), encoding="utf-8")

    tensors: Dict[str, torch.Tensor] = {}
    if pack.input_embeddings is not None:
        tensors["input_embeddings"] = pack.input_embeddings.detach().to(torch.float32).contiguous()
    if pack.output_embeddings is not None:
        tensors["output_embeddings"] = pack.output_embeddings.detach().to(torch.float32).contiguous()
    if tensors:
        try:
            from safetensors.torch import save_file

            save_file(tensors, str(out / TENSORS))
        except Exception:  # pragma: no cover - safetensors is a hard dep
            torch.save(tensors, out / "embeddings.pt")

    (out / ADDED_TOKENS).write_text(
        json.dumps(
            {e.candidate.surface: e.new_id for e in pack.entries if e.candidate.surface},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / CARD).write_text(model_card(pack), encoding="utf-8")
    log.info("wrote pack %s (%d tokens) to %s", pack.lang, len(pack), out)
    return out


def load_pack(path: str | Path, require_accepted: bool = True) -> VocabPack:
    """Read a pack from disk, refusing entries without an accepted certificate.

    Claim: bound — an uncertified token cannot enter a running model through
    this code path, which is the only code path the CLI and the server use.
    """
    p = Path(path)
    manifest = json.loads((p / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != "parity-vocab-pack/1":
        raise ValueError(f"unknown pack format {manifest.get('format')!r}")

    entries: List[PackEntry] = []
    dropped = 0
    keep_rows: List[int] = []
    for row, ed in enumerate(manifest["entries"]):
        entry = PackEntry.from_dict(ed)
        if require_accepted and not entry.certificate.accepted:
            dropped += 1
            continue
        entries.append(entry)
        keep_rows.append(row)
    if dropped:
        log.warning("dropped %d entries without an accepted certificate", dropped)

    inp = out = None
    tpath = p / TENSORS
    if tpath.exists():
        from safetensors.torch import load_file

        t = load_file(str(tpath))
        idx = torch.tensor(keep_rows, dtype=torch.long)
        inp = t["input_embeddings"].index_select(0, idx) if "input_embeddings" in t else None
        out = t["output_embeddings"].index_select(0, idx) if "output_embeddings" in t else None
    elif (p / "embeddings.pt").exists():  # pragma: no cover
        t = torch.load(p / "embeddings.pt")
        idx = torch.tensor(keep_rows, dtype=torch.long)
        inp = t.get("input_embeddings")
        out = t.get("output_embeddings")
        inp = inp.index_select(0, idx) if inp is not None else None
        out = out.index_select(0, idx) if out is not None else None

    return VocabPack(
        lang=manifest["lang"],
        base_model_id=manifest["base_model_id"],
        base_tokenizer_fingerprint=manifest.get("base_tokenizer_fingerprint", ""),
        base_vocab_size=int(manifest["base_vocab_size"]),
        entries=entries,
        input_embeddings=inp,
        output_embeddings=out,
        metadata=manifest.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def certificate_table(pack: VocabPack, statistic: str = "kl_next_token", n: int = 10) -> str:
    """Markdown table of the ``n`` highest-drift tokens in the pack.

    The *worst* tokens, not a random or best-case sample: a card should make a
    reader's job of finding the weakest point easy.

    Claim: bound.
    """
    rows = []
    for e in pack.entries:
        spec = e.certificate.bounds.get(statistic)
        if spec is None:
            continue
        rows.append((spec.value, e))
    rows.sort(key=lambda r: -r[0])
    lines = [
        "| token (surface) | base tokens | certified KL tail bound | mean KL | n calib |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for val, e in rows[:n]:
        spec = e.certificate.bounds[statistic]
        surface = e.candidate.surface.replace("|", "\\|").replace("\n", "\\n")
        lines.append(
            f"| `{surface}` | {e.candidate.length} | {val:.4g} | {spec.empirical_mean:.4g} | {spec.n_samples} |"
        )
    return "\n".join(lines)


def model_card(pack: VocabPack, extra: Optional[Dict[str, Any]] = None) -> str:
    """Generate the Hugging Face model card for a pack.

    The certified drift bound is not optional here — it is interpolated from the
    manifest, so the card cannot drift from the evidence.

    Claim: bound, reduction — the public artefact that states both what was
    gained and what was risked.
    """
    md = dict(pack.metadata)
    md.update(extra or {})
    lang = pack.lang
    worst_kl = pack.worst_bound("kl_next_token")
    worst_tv = pack.worst_bound("tv_next_token")
    n = len(pack)
    delta = alpha = 0.05
    for e in pack.entries:
        spec = e.certificate.bounds.get("kl_next_token")
        if spec:
            delta, alpha = spec.delta, spec.alpha
            break

    reduction = md.get("token_reduction")
    ratio_before = md.get("parity_ratio_before")
    ratio_after = md.get("parity_ratio_after")

    frontmatter = f"""---
license: apache-2.0
language:
- {lang}
base_model: {pack.base_model_id}
library_name: parity
tags:
- tokenizer
- vocabulary-adaptation
- multilingual
- inference-efficiency
- parity-vocab-pack
---"""

    return f"""{frontmatter}

# Parity vocabulary pack — `{lang}` for `{pack.base_model_id}`

**{n} new tokens. No continued pretraining. Certified drift.**

This pack adds {n} tokens to `{pack.base_model_id}` so that {lang} text costs
fewer tokens to read and write. The base model's weights are unchanged: the pack
only *appends* embedding rows, so a request that does not select this pack is
served by the original model, bit for bit.

Token cost in {lang} is a property of the base tokenizer — an artefact fit to a
corpus in which most of the world's writing systems were under-represented. This
pack repairs that artefact for one language. It says nothing about the language.

## What it buys

| metric | value |
| --- | --- |
| tokens saved on held-out {lang} text | {_pct(reduction)} |
| effective context gain | {(1 / (1 - reduction)) if isinstance(reduction, (int, float)) and reduction < 1 else float('nan'):.2f}x |
| tokens per English-equivalent sentence, before | {ratio_before if ratio_before is not None else 'n/a'} |
| tokens per English-equivalent sentence, after | {ratio_after if ratio_after is not None else 'n/a'} |
| new embedding rows | {n} |
| added parameters | {md.get('added_parameters', 'n/a')} |

## What it risks — the certificate

Every token in this pack carries a drift certificate measured on held-out
calibration contexts, disjoint from the ones its embedding was fitted on.

> With probability ≥ {1 - delta:.2f} over the calibration draw, at least
> {100 * (1 - alpha):.0f}% of future inputs from the calibration distribution
> have **KL(original ‖ Parity) ≤ {worst_kl:.4g} nats** and
> **total variation ≤ {worst_tv:.4g}**, for *every* token in this pack.

Those are the worst-case values across the pack; per-token bounds are in
`manifest.json`. Tokens whose bound exceeded the build tolerance were not
adopted.

**Scope of the guarantee.** These are finite-sample, distribution-free bounds
with respect to the calibration corpus ({md.get('calibration_corpus', 'FLORES-200')}).
They are *not* worst-case over all possible inputs. An adversarial prompt, or a
domain far from the calibration data, is outside the guarantee. English and
other non-pack languages are outside the guarantee in the other direction — they
are unaffected exactly, by construction, not statistically.

### Highest-drift tokens in this pack

{certificate_table(pack)}

## Use

```python
from parity import serving
router = serving.load("{pack.base_model_id}", packs=["<this-repo-id>"])
print(router.encode("...", view="{lang}"))   # fewer tokens
print(router.encode("...", view="base"))     # the original tokenizer, unchanged
```

## Build provenance

| field | value |
| --- | --- |
| base model | `{pack.base_model_id}` |
| base vocab size | {pack.base_vocab_size} |
| tokenizer fingerprint | `{pack.base_tokenizer_fingerprint[:16]}…` |
| mining corpus | {md.get('mining_corpus', 'n/a')} |
| calibration corpus | {md.get('calibration_corpus', 'n/a')} |
| synthesis solver | {md.get('solver', 'n/a')} |
| build FLOPs (measured) | {md.get('build_flops', 'n/a')} |
| build wall-clock (s) | {md.get('build_seconds', 'n/a')} |
| parity version | {md.get('parity_version', 'n/a')} |

## Contributing a pack for your language

See [`docs/contributing-a-pack.md`](https://github.com/) in the Parity
repository. In short: point the CLI at a corpus you trust for your language,
review the mined tokens (they are printed as strings, not ids), and open a pull
request with the resulting pack. Review of the token list by speakers of the
language is part of the process, not an optional extra.
"""


def export_added_tokens(pack: VocabPack, path: str | Path) -> Path:
    """Write a plain ``surface -> id`` map for tools outside this repo.

    Note the caveat recorded in the file: Parity tokens are defined over base
    *token id* sequences, so re-adding them to a Hugging Face tokenizer as
    strings is an approximation that can differ at surface boundaries.  The
    id-space definition in ``manifest.json`` is authoritative.

    Claim: infrastructure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "_note": (
                    "Parity tokens are defined as base-token-id sequences (see manifest.json). "
                    "This string map is provided for interoperability only; string-level "
                    "re-addition can differ at surface boundaries."
                ),
                "tokens": {e.candidate.surface: e.new_id for e in pack.entries},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return p
