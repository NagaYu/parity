"""Tables and Markdown for the benchmark results.

Every table carries the provenance of its numbers.  A results table that mixes
measured and extrapolated figures without saying which is which is the easiest
way to mislead a reader who is on your side, so the ``provenance`` column is not
optional here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CONDITION_LABELS = {
    "A": "(A) original model, original vocabulary",
    "B": "(B) vocabulary expansion + continued pretraining",
    "C": "(C) zero-shot tokenizer transfer",
    "D": "(D) Parity — synthesis + certificate + submodular selection",
}


def _fmt(x: Optional[float], spec: str = ".3f", none: str = "—") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return none
    return format(x, spec)


def results_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Markdown table of the per-condition, per-language results.

    Claim: reduction, non-regression — the primary results table, benchmark
    metrics (1), (2) and (5).
    """
    out = [
        "| lang | condition | tokens/sent | token reduction | eff. context | bits/char | Δ bits/char | retrieval acc | Δ acc | English Δ bits/char | certified KL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        red = r.get("token_reduction", 0.0) or 0.0
        gain = 1.0 / max(1e-9, 1.0 - red)
        out.append(
            f"| {r['lang']} | {r['condition']} | {_fmt(r.get('tokens_per_sentence'), '.1f')} | "
            f"{_fmt(100 * red, '.1f')}% | x{_fmt(gain, '.2f')} | "
            f"{_fmt(r.get('bits_per_character'), '.4f')} | {_fmt(r.get('bpc_delta'), '+.4f')} | "
            f"{_fmt(r.get('retrieval_accuracy'), '.3f')} | {_fmt(r.get('retrieval_delta'), '+.3f')} | "
            f"{_fmt(r.get('english_bpc_delta'), '+.6f')} | "
            f"{_fmt(r.get('certified_kl_bound'), '.4g')} |"
        )
    out.append("")
    out.append(
        "English is scored under the **base view**, which is how a router serves it. "
        "For (C) and (D) the column is exactly `+0.000000` — packs only append rows, so an "
        "English request is served by the original model bit for bit. For (B) it is non-zero "
        "in *either* direction, because continued pretraining rewrote the shared weights; the "
        "magnitude is what matters there, not the sign."
    )
    return "\n".join(out)


def cost_table(cost: Dict[str, Any]) -> str:
    """Markdown table of the cost comparison, provenance included.

    Claim: low-cost — benchmark metric (4).
    """
    lines = [
        "| item | FLOPs | wall-clock (s) | provenance |",
        "| --- | ---: | ---: | --- |",
    ]
    for stage in cost["parity"]["stages"]:
        lines.append(
            f"| Parity — {stage['stage']} | {stage['flops']:.3e} | {stage['seconds']:.1f} | measured |"
        )
    lines.append(f"| **Parity — total** | **{cost['parity']['flops']:.3e}** | {cost['parity']['seconds']:.1f} | measured |")
    m = cost["continued_pretraining_measured"]
    if m["tokens"]:
        lines.append(
            f"| continued pretraining ({m['tokens']:,} tokens, as run here) | {m['flops']:.3e} | {m['seconds']:.1f} | measured |"
        )
    for ref in cost["continued_pretraining_reference"]:
        lines.append(
            f"| continued pretraining ({ref['tokens'] / 1e9:.0f}B tokens, published scale) | {ref['flops']:.3e} | — | **extrapolated** |"
        )
    lines.append("")
    lines.append(
        f"Parity is **{cost['ratio_vs_reference_1b']:.0f}x–{cost['ratio_vs_reference_10b']:.0f}x** cheaper than the "
        f"published-scale baseline (~{cost['orders_of_magnitude_conservative']:.1f} orders of magnitude, "
        f"quoted at the conservative end), and adds "
        f"{100 * cost['added_param_fraction']:.3f}% to the parameter count."
    )
    return "\n".join(lines)


def serving_table(serving: Dict[str, Any]) -> str:
    """Markdown table of multi-tokenizer serving overhead.

    Claim: low-cost — benchmark metric (6).
    """
    multi, base = serving["multi_view"], serving["base_only"]
    return "\n".join(
        [
            "| workload | requests | tokens processed | tokens/s | dispatch overhead |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| all requests on the base view | {base['n_requests']} | {base['total_tokens']} | "
            f"{base['tokens_per_second']:.0f} | {100 * base['dispatch_overhead']:.2f}% |",
            f"| mixed views (packs active) | {multi['n_requests']} | {multi['total_tokens']} | "
            f"{multi['tokens_per_second']:.0f} | {100 * multi['dispatch_overhead']:.2f}% |",
            "",
            f"The mixed-view run processes {100 * (1 - serving['token_ratio']):.1f}% **fewer** tokens for the same "
            f"text and finishes {serving['effective_speedup']:.2f}x faster in wall-clock. View dispatch "
            f"(tokenizer choice + logit mask) accounts for {100 * multi['dispatch_overhead']:.2f}% of that time; "
            "the model-side work is unchanged because all views share one id space and one weight set.",
        ]
    )


def certificate_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Markdown table of certified drift vs. realised coverage, per language.

    The right check for a ``(1−α, 1−δ)`` tolerance limit is **coverage** — the
    share of fresh measurements at or below the bound — not the fresh maximum.
    The bound is explicitly allowed an ``α`` tail, so a maximum above it is
    expected behaviour, and putting that maximum next to the bound in a table
    invites exactly the wrong conclusion. The observed maximum is still shown,
    in its own column, labelled as what it is.

    Claim: bound — benchmark metric (3).
    """
    lines = [
        "| lang | tokens | certified KL (worst) | fresh coverage | target | under-covering | verdict | acceptance rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['lang']} | {r['n_tokens']} | {_fmt(r['certified_kl'], '.4g')} | "
            f"{_fmt(100 * r.get('mean_coverage', float('nan')), '.1f')}% | "
            f"{_fmt(100 * r.get('mean_target', float('nan')), '.0f')}% | "
            f"{_fmt(100 * r.get('violation_rate', float('nan')), '.1f')}% "
            f"(≤{_fmt(100 * r.get('violation_allowance', float('nan')), '.0f')}% allowed) | "
            f"{'PASS' if r['inside'] else '**FAIL**'} | "
            f"{_fmt(100 * r['acceptance_rate'], '.1f')}% |"
        )
    lines.append("")
    lines.append(
        "Coverage is measured on held-out contexts the certificate never saw. "
        "'Under-covering' counts bounds whose realised coverage fell below target; the "
        "guarantee permits a δ fraction of them to, which is the allowance in brackets."
    )
    worst_max = max((r.get("measured_kl", 0.0) for r in rows), default=0.0)
    lines.append(
        f"Largest single fresh KL measurement across all languages: {worst_max:.4g} nats — above the "
        "tail bound, as a tail bound permits, and reported so the tail is visible rather than implied."
    )
    return "\n".join(lines)


def write_report(path: str | Path, sections: Dict[str, str], meta: Dict[str, Any]) -> Path:
    """Write the Markdown results report.

    Claim: infrastructure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Parity benchmark results",
        "",
        f"- model: `{meta.get('model')}`",
        f"- corpus: `{meta.get('corpus_source')}` ({meta.get('n_sentences')} aligned sentences)",
        f"- languages: {', '.join(meta.get('langs', []))}",
        f"- budget: {meta.get('budget')} embedding rows per language",
        f"- tolerance: KL <= {meta.get('max_kl')} nats at ({1 - meta.get('alpha', 0.05):.2f}, {1 - meta.get('delta', 0.05):.2f})",
        f"- parity version: {meta.get('parity_version')}",
        "",
    ]
    if str(meta.get("corpus_source", "")).startswith("fixture") or str(meta.get("corpus_source", "")) == "embedded_sample":
        header += [
            "> **These numbers are illustrative, not evidence.** The run used the offline",
            "> fixture corpus (and possibly the fixture model). Re-run against FLORES-200",
            "> and a real checkpoint before quoting anything here.",
            "",
        ]
    body = "\n".join(header)
    for title, content in sections.items():
        body += f"\n## {title}\n\n{content}\n"
    p.write_text(body, encoding="utf-8")
    return p


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Write the machine-readable results blob.

    Claim: infrastructure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p
