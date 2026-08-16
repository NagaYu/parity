"""Build the language x tokenizer fertility atlas and push it as a HF Dataset.

    python scripts/build_atlas.py --tokenizers Qwen/Qwen2.5-0.5B-Instruct,meta-llama/Llama-3.2-1B \
        --langs ja,hi,ar,th,sw,bn,te,am,my,ko --out data/atlas.jsonl
    python scripts/build_atlas.py ... --push your-org/parity-fertility-atlas

One row per (tokenizer, language): tokens per character, tokens per word where
the script has words, and the number that matters — ``parity_ratio``, the tokens
this language costs per English-equivalent sentence on a parallel corpus.

The atlas is a measurement, and measurement is the *smallest* part of this
project. It exists to make the baseline auditable and to let a language
community see where their language sits before deciding whether a pack is worth
building. It is not the deliverable; `parity build` is.

Framing note, enforced in the emitted dataset card: a high ratio is a property
of the tokenizer, not of the language. The dataset carries that sentence with it
so a row lifted out of context still says what it means.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# Allow both `python scripts/x.py` and `python -m scripts.x`: the former puts
# scripts/ on sys.path rather than the repo root.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("parity.atlas")

DATASET_CARD = """---
license: cc-by-sa-4.0
task_categories:
- text-generation
language:
- multilingual
tags:
- tokenizer
- fertility
- multilingual
- parity
pretty_name: Parity fertility atlas
---

# Parity fertility atlas

How many tokens each tokenizer charges for the same meaning, measured on a
parallel corpus ({corpus}).

| column | meaning |
| --- | --- |
| `tokenizer_id` | the tokenizer measured |
| `lang` | ISO code |
| `tokens_per_char` | tokens per NFC character, excluding whitespace |
| `tokens_per_word` | tokens per whitespace word; `null` for scripts without word spaces |
| `parity_ratio` | tokens(target) / tokens(aligned English) — **the headline** |
| `parity_ratio_median` | median of the per-sentence ratios |
| `effective_context_fraction` | `1 / parity_ratio`: the share of the advertised context window a speaker of this language actually receives |

## How to read this

A `parity_ratio` of 2.4 means a speaker of that language spends 2.4x the tokens
to say the same thing, gets 42% of the context window, and pays 2.4x per
message.

**This is a property of the tokenizer, not of the language.** A tokenizer is
fitted to a corpus; the corpora used to fit today's tokenizers under-represent
most of the world's writing systems, and the ratios below are the arithmetic
consequence. Nothing here says any language is inefficient, verbose, or harder
to model. The measurement is of an engineering artefact, and the artefact is
repairable — that is what the [Parity](https://github.com/) repository is for.

## Reproducing

```bash
python scripts/build_atlas.py --tokenizers {tokenizers} --langs {langs}
```

Rows are produced by `parity.fertility.measure_fertility` on {n} aligned
sentences from {corpus}.
"""


def build_rows(
    tokenizer_ids: Sequence[str],
    langs: Sequence[str],
    split: str = "devtest",
    max_sentences: Optional[int] = None,
    offline: bool = False,
) -> List[Dict[str, Any]]:
    """Measure every (tokenizer, language) pair and return atlas rows.

    Claim: reduction — establishes the baseline that every reduction number in
    this project is relative to.
    """
    from parity.corpora import load_parallel
    from parity.fertility import fertility_table
    from parity.tokenization import AugmentedTokenizer, HFTokenizer

    corpus = load_parallel(list(langs), split=split, max_sentences=max_sentences, allow_download=not offline)
    log.info("corpus: %s, %d aligned sentences", corpus.source, corpus.n)

    rows: List[Dict[str, Any]] = []
    for tid in tokenizer_ids:
        try:
            if tid == "tiny":
                from parity.corpora import load_embedded_sample
                from parity.tiny import build_tiny_tokenizer

                base = build_tiny_tokenizer(load_embedded_sample().by_lang["en"] * 8, vocab_size=900)
            else:
                base = HFTokenizer.from_pretrained(tid)
        except Exception as exc:
            log.warning("skipping tokenizer %s: %s", tid, exc)
            continue
        aug = AugmentedTokenizer(base)
        table = fertility_table(aug, corpus, list(langs) + ["en"], tokenizer_id=tid)
        for lang, rep in table.items():
            row = rep.to_dict()
            row["corpus"] = corpus.source
            row["split"] = corpus.split
            rows.append(row)
        log.info("measured %s over %d languages", tid, len(table))
    return rows


def write_jsonl(rows: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    """Write the atlas as JSONL.

    Claim: infrastructure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    """Print the atlas, sorted by how badly each language is served.

    Claim: reduction.
    """
    by_tok: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_tok.setdefault(r["tokenizer_id"], []).append(r)
    for tid, group in by_tok.items():
        print(f"\n{tid}")
        print(f"{'lang':<6}{'tokens/char':>13}{'tokens/word':>13}{'vs English':>12}{'eff. context':>14}")
        print("-" * 58)
        for r in sorted(group, key=lambda r: -(r.get("parity_ratio") or 0)):
            tpw = r.get("tokens_per_word")
            pr = r.get("parity_ratio")
            ec = r.get("effective_context_fraction")
            print(
                f"{r['lang']:<6}{r['tokens_per_char']:>13.3f}"
                f"{(f'{tpw:.3f}' if tpw else '—'):>13}"
                f"{(f'{pr:.2f}x' if pr else '1.00x'):>12}"
                f"{(f'{100 * ec:.0f}%' if ec else '100%'):>14}"
            )


def push(rows: Sequence[Dict[str, Any]], repo_id: str, card: str, private: bool = False) -> str:
    """Push the atlas to the Hub as a Dataset, card included.

    Claim: infrastructure — the atlas is only useful if the people it describes
    can find it.
    """
    from datasets import Dataset
    from huggingface_hub import HfApi

    ds = Dataset.from_list(list(rows))
    ds.push_to_hub(repo_id, private=private)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("pushed %d rows to https://huggingface.co/datasets/%s", len(rows), repo_id)
    return f"https://huggingface.co/datasets/{repo_id}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Build the Parity fertility atlas")
    ap.add_argument("--tokenizers", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--langs", default="ja,hi,ar,th,sw")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--out", default="data/atlas.jsonl")
    ap.add_argument("--push", default=None, help="HF dataset repo id, e.g. your-org/parity-fertility-atlas")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    tokenizers = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    rows = build_rows(tokenizers, langs, args.split, args.max_sentences, args.offline)
    if not rows:
        print("no rows produced; check --tokenizers and network access")
        return 1

    print_table(rows)
    path = write_jsonl(rows, args.out)
    print(f"\nwrote {path} ({len(rows)} rows)")

    if args.push:
        card = DATASET_CARD.format(
            corpus=rows[0].get("corpus", "FLORES-200"),
            tokenizers=",".join(tokenizers),
            langs=",".join(langs),
            n=rows[0].get("n_sentences", "?"),
        )
        print(push(rows, args.push, card, args.private))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
