"""The headline figure: token reduction against downstream quality.

    python figures/make_pareto.py runs/latest/results.json figures/pareto.png

What the reader should be able to see at a glance:

* **x** — tokens removed from the same text (higher is better),
* **y** — downstream quality relative to the untouched model (1.0 = unchanged),
* **bubble area** — build cost in FLOPs, on a log scale,
* (D) sitting in the high-reduction / no-degradation corner with a *small*
  bubble, next to (B) in roughly the same corner with a bubble orders of
  magnitude larger.

Two honesty rules are enforced in the drawing code, not left to the caption:

1. Any point whose provenance is not ``measured`` is hatched and labelled.
2. If the run used the offline fixture corpus or fixture model, the whole figure
   gets an "ILLUSTRATIVE" watermark. A pretty plot travels further than its
   caption, so the caveat has to be inside the image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

CONDITION_STYLE = {
    "A": {"label": "(A) original model + original vocabulary", "color": "#6b7280", "marker": "o"},
    "B": {"label": "(B) vocab expansion + continued pretraining", "color": "#b45309", "marker": "s"},
    "C": {"label": "(C) zero-shot tokenizer transfer", "color": "#0e7490", "marker": "^"},
    "D": {"label": "(D) Parity — synthesis + certificate + selection", "color": "#7c3aed", "marker": "*"},
}


def _bubble(flops: float) -> float:
    """Map build FLOPs to a marker area, log-scaled and floored so 0 is visible.

    Claim: low-cost — the cost claim is only legible if it is *drawn*, and a
    linear scale would render Parity's bubble invisible next to the baseline's.
    """
    if not flops or flops <= 0:
        return 60.0
    return 40.0 + 40.0 * max(0.0, math.log10(flops) - 8.0)


def render(payload: Dict[str, Any], out_path: str | Path, dpi: int = 160) -> Path:
    """Draw the Pareto figure from a benchmark ``results.json`` payload.

    Claim: reduction, non-regression, low-cost — the single image that carries
    all three, and marks the provenance of every point it draws.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = payload.get("meta", {})
    languages = [l for l in payload.get("languages", []) if "conditions" in l]
    illustrative = bool(meta.get("demo")) or str(meta.get("corpus_source", "")).startswith(("fixture", "embedded"))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.55, 1.0]}, constrained_layout=True
    )

    lang_markers: Dict[str, str] = {}
    for lang_result in languages:
        lang = lang_result["lang"]
        for cond in lang_result["conditions"]:
            key = cond["condition"]
            style = CONDITION_STYLE.get(key)
            if style is None:
                continue
            x = 100.0 * (cond.get("token_reduction") or 0.0)
            y = cond.get("quality_retention")
            if y is None or y != y:
                continue
            extrapolated = cond.get("provenance", "measured") != "measured"
            ax.scatter(
                x,
                y,
                s=_bubble(cond.get("build_flops", 0.0)),
                c=style["color"],
                marker=style["marker"],
                alpha=0.85,
                edgecolors="white" if not extrapolated else "black",
                linewidths=1.2,
                hatch="///" if extrapolated else None,
                zorder=3,
            )
            ax.annotate(
                f"{lang}·{key}",
                (x, y),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=8,
                color=style["color"],
            )
            lang_markers[lang] = style["marker"]

    ax.axhline(1.0, color="#9ca3af", lw=1, ls="--", zorder=1)
    ax.annotate(
        "no downstream degradation",
        (ax.get_xlim()[0], 1.0),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=8,
        color="#6b7280",
    )
    ax.set_xlabel("tokens removed from the same text  (%)  →  cheaper, longer effective context")
    ax.set_ylabel("downstream quality retained  (1.0 = original model)")
    ax.set_title("Reduction vs. quality — bubble area ∝ log build FLOPs", fontsize=11)
    ax.grid(alpha=0.25, zorder=0)

    handles = [
        Line2D([], [], color=s["color"], marker=s["marker"], ls="", ms=9, label=s["label"])
        for s in CONDITION_STYLE.values()
    ]
    handles.append(Line2D([], [], color="black", marker="o", ls="", ms=9, mfc="none", label="hatched = extrapolated"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)

    # -- right panel: cost, log scale ---------------------------------------
    costs = [l["cost"] for l in languages if "cost" in l]
    if costs:
        c = costs[0]
        labels, values, colors, hatches = [], [], [], []
        labels.append("(D) Parity build")
        values.append(max(1.0, c["parity"]["flops"]))
        colors.append(CONDITION_STYLE["D"]["color"])
        hatches.append(None)
        m = c["continued_pretraining_measured"]
        if m["flops"]:
            labels.append(f"(B) CPT as run\n({m['tokens']:,} tok)")
            values.append(m["flops"])
            colors.append(CONDITION_STYLE["B"]["color"])
            hatches.append(None)
        for ref in c["continued_pretraining_reference"]:
            labels.append(f"(B) CPT at\n{ref['tokens'] / 1e9:.0f}B tokens")
            values.append(ref["flops"])
            colors.append(CONDITION_STYLE["B"]["color"])
            hatches.append("///")
        bars = ax2.bar(range(len(values)), values, color=colors, alpha=0.85)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("black")
        ax2.set_yscale("log")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel("build FLOPs (log scale)")
        ax2.set_title(
            f"Build cost — Parity is {c['ratio_vs_reference_1b']:.0f}x–{c['ratio_vs_reference_10b']:.0f}x cheaper",
            fontsize=11,
        )
        ax2.grid(axis="y", alpha=0.25)
        ax2.annotate(
            "hatched = extrapolated from 6·N·tokens,\nnot executed here",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            fontsize=7.5,
            color="#374151",
        )

    subtitle = (
        f"{meta.get('model', '?')} · {meta.get('corpus_source', '?')} · "
        f"budget {meta.get('budget', '?')} rows/language · "
        f"certified KL ≤ {meta.get('max_kl', '?')} nats at "
        f"({1 - float(meta.get('alpha', 0.05)):.2f}, {1 - float(meta.get('delta', 0.05)):.2f})"
    )
    fig.suptitle("Parity: certified vocabulary augmentation without continued pretraining\n" + subtitle, fontsize=12)

    if illustrative:
        fig.text(
            0.5,
            0.5,
            "ILLUSTRATIVE — fixture model/corpus",
            fontsize=34,
            color="#dc2626",
            alpha=0.16,
            ha="center",
            va="center",
            rotation=22,
            zorder=10,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Render the Parity Pareto figure")
    ap.add_argument("results", help="path to results.json from benchmarks.run")
    ap.add_argument("out", nargs="?", default="figures/pareto.png")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    path = render(payload, args.out, dpi=args.dpi)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
