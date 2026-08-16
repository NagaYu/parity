"""Parity — Gradio Space.

Type in your language. See what it costs you, and what a Parity pack gives back.

The Space runs the **tokenizer-only** path: no model weights, no GPU, CPU
cold-start in seconds. That is enough to show token cost, the ratio against
English, the share of the context window a speaker actually receives, and the
reduction a pack delivers — because all of those are properties of tokenization.
The behavioural guarantee is a property of the weights and is shown here as the
certificate the pack shipped with, not recomputed in the browser.

Framing, which is part of the specification and not decoration: the numbers
below describe a *tokenizer*, an artefact fitted to a corpus. They do not
describe a language. Nothing in this app calls a language inefficient, because
no language is.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from parity.corpora import LANGUAGES, count_chars, load_embedded_sample, normalize
from parity.tokenization import AugmentedTokenizer, BASE_VIEW, diff_tokenization

log = logging.getLogger("parity.app")

DEFAULT_MODEL = os.environ.get("PARITY_MODEL", "HuggingFaceTB/SmolLM2-135M")
PACK_DIR = Path(os.environ.get("PARITY_PACKS", "packs"))
#: Comma-separated Hub model repo ids holding Parity packs, e.g.
#: ``NagaYu/parity-ja-smollm2-135m``.  Set on the Space so it picks up published
#: packs without them being vendored into the repo.
PACK_REPOS = [r.strip() for r in os.environ.get("PARITY_PACK_REPOS", "").split(",") if r.strip()]
CONTEXT_WINDOW = 128_000

SAMPLE = load_embedded_sample()
DEMO_LANGS = ["ja", "hi", "ar", "th", "sw"]

_STATE: Dict[str, Any] = {"tokenizer": None, "aug": None, "packs": {}, "model_id": None, "error": None}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _pack_dirs(notes: List[str]) -> List[Path]:
    """Local pack directories plus any downloaded from the Hub.

    The Space ships no pack in its own repo — packs are Model repos, and
    vendoring a copy would let the two drift apart. ``PARITY_PACK_REPOS`` names
    them and they are fetched at startup.

    Claim: infrastructure — one published artefact, referenced, not duplicated.
    """
    dirs: List[Path] = []
    if PACK_DIR.exists():
        dirs += [d for d in sorted(PACK_DIR.iterdir()) if (d / "manifest.json").exists()]
    for repo in PACK_REPOS:
        try:
            from huggingface_hub import snapshot_download

            dirs.append(Path(snapshot_download(repo_id=repo, repo_type="model")))
        except Exception as exc:
            notes.append(f"could not fetch pack `{repo}`: {type(exc).__name__}")
    return dirs


def load_backend(model_id: str) -> Tuple[Optional[AugmentedTokenizer], str]:
    """Load a tokenizer and any packs found on disk; return ``(aug, status)``.

    Falls back to the offline fixture tokenizer if the Hub is unreachable, and
    says so in the status line rather than silently showing fixture numbers as
    if they were real.

    Claim: infrastructure, reduction.
    """
    if _STATE["aug"] is not None and _STATE["model_id"] == model_id:
        return _STATE["aug"], _STATE.get("status", "")

    notes: List[str] = []
    try:
        from parity.tokenization import HFTokenizer

        base = HFTokenizer.from_pretrained(model_id)
        notes.append(f"tokenizer: `{model_id}` ({base.vocab_size:,} tokens)")
    except Exception as exc:  # offline or gated model
        from parity.tiny import build_tiny_tokenizer

        base = build_tiny_tokenizer(SAMPLE.by_lang["en"] * 8, vocab_size=900)
        notes.append(
            f"⚠️ could not load `{model_id}` ({type(exc).__name__}); using the offline **fixture** "
            "tokenizer. Ratios below are illustrative, not measurements of a production tokenizer."
        )

    aug = AugmentedTokenizer(base)
    loaded: Dict[str, Any] = {}
    for d in _pack_dirs(notes):
        from parity.pack import load_pack

        try:
            pack = load_pack(d)
            aug.attach(pack)
            loaded[pack.lang] = pack
        except Exception as exc:
            # Usually a fingerprint mismatch: the pack was built for a different
            # tokenizer. Saying which is more useful than silently ignoring it.
            notes.append(f"skipped pack `{Path(d).name}`: {exc}")
    if loaded:
        aug.check_invariants()
        notes.append("packs: " + ", ".join(f"`{k}` ({len(v)} tokens)" for k, v in loaded.items()))
    else:
        notes.append(
            "no vocabulary packs found — showing the **baseline** only. "
            "Build one with `parity build --model … --lang ja --budget 8000` and drop it in `packs/`."
        )

    _STATE.update({"tokenizer": base, "aug": aug, "packs": loaded, "model_id": model_id, "status": "  \n".join(notes)})
    return aug, _STATE["status"]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _english_reference(aug: AugmentedTokenizer, lang: str, text: str) -> Tuple[float, str]:
    """Estimate the English-equivalent token count for ``text``.

    If the input is one of the sample sentences we use its actual translation,
    which is exact.  Otherwise we scale by the corpus-level tokens-per-character
    ratio measured on the embedded parallel sample — an estimate, and labelled
    as one in the UI.

    Claim: reduction — the ratio is the number a user feels, so it must be
    computed from meaning-matched text wherever possible.
    """
    text_n = normalize(text)
    for i, s in enumerate(SAMPLE.by_lang.get(lang, [])):
        if normalize(s) == text_n:
            return float(len(aug.encode(SAMPLE.by_lang["en"][i], BASE_VIEW))), "exact (aligned translation)"

    en_tokens = sum(len(aug.encode(s, BASE_VIEW)) for s in SAMPLE.by_lang["en"])
    en_chars = sum(count_chars(s) for s in SAMPLE.by_lang["en"])
    tgt_tokens = sum(len(aug.encode(s, BASE_VIEW)) for s in SAMPLE.by_lang.get(lang, SAMPLE.by_lang["en"]))
    tgt_chars = sum(count_chars(s) for s in SAMPLE.by_lang.get(lang, SAMPLE.by_lang["en"]))
    if not tgt_chars or not en_chars or not en_tokens:
        return 1.0, "unavailable"
    # Two corpus-level constants from the aligned sample:
    #   how many characters this language needs per English character, and
    #   how many characters the tokenizer packs into one English token.
    chars_ratio = en_chars / tgt_chars
    chars_per_en_token = en_chars / en_tokens
    est = count_chars(text_n) * chars_ratio / chars_per_en_token
    return max(1.0, est), "estimated from the sample corpus"


def analyse(text: str, lang: str, model_id: str) -> Tuple[str, str, str, str]:
    """Measure one input and render the four output panels.

    Claim: reduction — this is the demonstration: the token premium, the
    context-window loss, and what a pack returns.
    """
    aug, status = load_backend(model_id)
    if not text.strip():
        return status, "_Type something above._", "", ""

    text = normalize(text)
    base_ids = aug.encode(text, BASE_VIEW)
    n_base = len(base_ids)
    chars = count_chars(text)

    en_equiv, provenance = _english_reference(aug, lang, text)
    ratio = n_base / max(1e-9, en_equiv)

    pack = _STATE["packs"].get(lang)
    has_pack = pack is not None
    view = aug.view(lang) if has_pack else BASE_VIEW
    diff = diff_tokenization(aug, text, view)
    n_after = len(diff.aug_ids)
    reduction = diff.reduction

    spec = LANGUAGES.get(lang)
    name = f"{spec.endonym} ({spec.name})" if spec else lang

    # -- panel 1: the premium ----------------------------------------------
    eff_before = CONTEXT_WINDOW / max(1e-9, ratio)
    ratio_after = ratio * (1 - reduction)
    eff_after = CONTEXT_WINDOW / max(1e-9, ratio_after)
    summary = f"""
### {name}

| | tokens | vs. English | effective 128k window |
| --- | ---: | ---: | ---: |
| **now** | {n_base} | **{ratio:.2f}x** | {eff_before:,.0f} tokens ({100 / max(1e-9, ratio):.0f}%) |
"""
    if has_pack:
        summary += (
            f"| **with the `{lang}` pack** | {n_after} | **{ratio_after:.2f}x** | "
            f"{eff_after:,.0f} tokens ({100 / max(1e-9, ratio_after):.0f}%) |\n"
        )
        summary += f"""
**−{100 * reduction:.1f}% tokens** on this input → **{1 / max(1e-9, 1 - reduction):.2f}x** more usable context,
and the same fraction off the per-message inference bill.
"""
    else:
        summary += f"""
No pack loaded for `{lang}`, so only the baseline is shown.

```bash
parity build --model {model_id} --lang {lang} --budget 8000
```
"""
    summary += (
        f"\n<sub>{chars} characters · English-equivalent length {provenance}. "
        "Token cost is a property of the tokenizer — an artefact fitted to a corpus that "
        "under-represented most writing systems — not of the language.</sub>"
    )

    # -- panel 2: the tokens ------------------------------------------------
    def chips(pieces: List[str], mark_from: Optional[int] = None, ids: Optional[List[int]] = None) -> str:
        out = []
        for i, p in enumerate(pieces):
            shown = (p or "␣").replace("<", "&lt;").replace(">", "&gt;").replace("Ġ", "␣").replace("\n", "⏎")
            is_new = ids is not None and mark_from is not None and ids[i] >= mark_from
            colour = "#ede9fe" if is_new else "#f1f5f9"
            border = "#7c3aed" if is_new else "#cbd5e1"
            out.append(
                f'<span style="display:inline-block;margin:2px;padding:2px 6px;border-radius:5px;'
                f'background:{colour};border:1px solid {border};font-family:ui-monospace,monospace;'
                f'font-size:13px">{shown}</span>'
            )
        return "".join(out)

    tokens_html = f"""
<div style="margin-bottom:10px"><b>Original tokenizer — {n_base} tokens</b><br>{chips(diff.base_pieces)}</div>
"""
    if has_pack:
        tokens_html += f"""
<div><b>With the <code>{lang}</code> pack — {n_after} tokens</b>
<span style="color:#6b7280">(purple = a Parity token)</span><br>
{chips(diff.aug_pieces, aug.base_vocab_size, diff.aug_ids)}</div>
"""

    # -- panel 3: the certificate ------------------------------------------
    if has_pack:
        kl = pack.worst_bound("kl_next_token")
        tv = pack.worst_bound("tv_next_token")
        off = pack.worst_bound("offcontext_mass")
        spec0 = next((e.certificate.bounds.get("kl_next_token") for e in pack.entries), None)
        alpha = spec0.alpha if spec0 else 0.05
        delta = spec0.delta if spec0 else 0.05
        cert = f"""
### What this pack promises

> With probability ≥ {1 - delta:.2f} over the calibration draw, at least
> {100 * (1 - alpha):.0f}% of inputs from the calibration distribution move the
> next-token distribution by at most **{kl:.4g} nats** (KL) and **{tv:.4g}**
> (total variation) — for every one of the {len(pack)} tokens in this pack.

- Off-context firing: ≤ **{off:.3g}** probability where the token does not belong.
- English and every other language: **exactly unchanged**. Packs only *append*
  embedding rows, and a request on the base view cannot see or emit a pack token.
- Not covered: adversarial prompts, and domains far from the calibration corpus.
  These are finite-sample, distribution-free bounds, not worst-case ones.

Build cost: {pack.metadata.get('build_flops', 'n/a')} FLOPs — no continued pretraining.
"""
    else:
        cert = (
            "### What a pack would promise\n\n"
            "Every token ships with a measured drift certificate, and tokens whose bound "
            "exceeds the tolerance are not adopted. Load a pack to see the numbers."
        )

    return status, summary, tokens_html, cert


def compare_all(model_id: str) -> str:
    """Rank the demo languages by what the tokenizer charges them.

    Claim: reduction — the cross-language view, which is where the imbalance
    stops looking like an anecdote.
    """
    aug, _ = load_backend(model_id)
    en_tokens = sum(len(aug.encode(s, BASE_VIEW)) for s in SAMPLE.by_lang["en"])
    rows = ["| language | tokens (same 24 sentences) | vs. English | effective 128k window | with a pack |",
            "| --- | ---: | ---: | ---: | ---: |"]
    entries = []
    for lang in ["en"] + DEMO_LANGS:
        sents = SAMPLE.by_lang.get(lang)
        if not sents:
            continue
        n = sum(len(aug.encode(s, BASE_VIEW)) for s in sents)
        ratio = n / max(1, en_tokens)
        pack = _STATE["packs"].get(lang)
        if pack is not None:
            view = aug.view(lang)
            n_after = sum(len(aug.encode(s, view)) for s in sents)
            after = f"**{n_after / max(1, en_tokens):.2f}x** (−{100 * (1 - n_after / max(1, n)):.0f}%)"
        else:
            after = "—"
        spec = LANGUAGES.get(lang)
        label = f"{spec.endonym} ({lang})" if spec else lang
        entries.append((ratio, f"| {label} | {n} | **{ratio:.2f}x** | {100 / max(1e-9, ratio):.0f}% | {after} |"))
    for _, row in sorted(entries, key=lambda r: -r[0]):
        rows.append(row)
    return "\n".join(rows) + (
        "\n\n<sub>Measured on the 24 aligned sentences embedded in this repository, so the comparison "
        "holds meaning constant. Run `scripts/build_atlas.py` for FLORES-200 numbers across more "
        "languages and tokenizers.</sub>"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface.

    Claim: infrastructure, reduction.
    """
    lang_choices = [(f"{LANGUAGES[l].endonym} — {LANGUAGES[l].name}", l) for l in DEMO_LANGS if l in LANGUAGES]

    with gr.Blocks(title="Parity — what your language costs, and how to get it back") as demo:
        gr.Markdown(
            """
# Parity

**Non-English text costs more tokens to say the same thing. That is a property of the
tokenizer, not of the language — and it is repairable without retraining the model.**

Type below in your language. You will see what you are charged now, how much of the
context window you actually receive, and what a Parity vocabulary pack returns —
with the certified bound on how far the model's behaviour may move.
"""
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Your text",
                    lines=4,
                    value=SAMPLE.by_lang["ja"][3],
                    placeholder="Type or paste text in your language…",
                )
            with gr.Column(scale=1):
                lang = gr.Dropdown(choices=lang_choices, value="ja", label="Language")
                model = gr.Textbox(value=DEFAULT_MODEL, label="Tokenizer / model")
                go = gr.Button("Measure", variant="primary")

        with gr.Row():
            examples = gr.Examples(
                examples=[[SAMPLE.by_lang[l][i], l] for l in DEMO_LANGS for i in (0, 3)],
                inputs=[text, lang],
                label="Try these (all say the same thing)",
            )

        status = gr.Markdown()
        summary = gr.Markdown()
        tokens = gr.HTML()
        with gr.Accordion("The certificate — what changes, and by how much", open=True):
            cert = gr.Markdown()
        with gr.Accordion("Every language, side by side", open=False):
            table = gr.Markdown()
            refresh = gr.Button("Compare all languages")

        gr.Markdown(
            """
---
**Scope.** Parity applies to open-weight models and to the providers that serve them:
it needs access to the embedding matrix. It cannot be applied from outside a closed API.

**How to add your language.** See `docs/contributing-a-pack.md`. You point the CLI at a
corpus you trust, review the mined tokens — they are printed as strings, not ids — and
open a pull request. Review by speakers of the language is part of the process.
"""
        )

        go.click(analyse, [text, lang, model], [status, summary, tokens, cert])
        text.submit(analyse, [text, lang, model], [status, summary, tokens, cert])
        lang.change(analyse, [text, lang, model], [status, summary, tokens, cert])
        refresh.click(compare_all, [model], [table])
        demo.load(analyse, [text, lang, model], [status, summary, tokens, cert])
        demo.load(compare_all, [model], [table])
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    build_ui().launch()
