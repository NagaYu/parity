"""Run the four-condition benchmark and write results, tables and the figure.

    python -m benchmarks.run --demo                     # offline fixture, ~1 min
    python -m benchmarks.run --model Qwen/Qwen2.5-0.5B-Instruct \
        --langs ja,hi,ar,th,sw --budget 4000 --out runs/qwen05b

Conditions, all sharing the **same selected token set** so that the comparison
isolates how the embeddings were obtained rather than which tokens got lucky:

(A) original model, original vocabulary
(B) same tokens, composition init, then continued pretraining (weights change)
(C) same tokens, mean-of-sub-tokens init, no optimisation, no certificate
(D) Parity: subspace least squares + certificate + submodular selection

Metrics, mapped to the project's list:

(1) fertility reduction ......... ``token_reduction`` per language
(2) downstream retention ........ ``bits_per_character`` and retrieval accuracy,
                                  plus the English columns for non-regression
(3) drift inside the certificate . re-verification on fresh contexts
(4) build cost .................. measured FLOPs vs (B), itemised
(5) effective context / cost .... ``context_gain``, ``cost_reduction``
(6) multi-tokenizer overhead .... serving throughput, mixed vs single view
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from benchmarks import cost as costmod
from benchmarks import report as reportmod
from benchmarks.tasks import QualityReport, evaluate_condition
from parity import _version
from parity.adapters import TorchLMAdapter
from parity.baselines import TrainingConfig, continued_pretraining, pack_from_tokens, zero_shot_transfer_config
from parity.build import BuildConfig, attach_and_verify, build_pack, negative_lines, split_corpus
from parity.certificate import CertifierConfig, DriftCertifier, verify_certificates
from parity.corpora import DEFAULT_TARGETS, Corpus, ParallelCorpus, expand_for_testing, load_embedded_sample, load_parallel
from parity.miner import MinerConfig
from parity.selection import SelectionConfig
from parity.serving import MultiTokenizerRouter, Request
from parity.synthesis import CalibrationIndex, EmbeddingSynthesizer, SynthesisConfig
from parity.tokenization import AugmentedTokenizer

log = logging.getLogger("benchmarks.run")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def make_factory(model_id: str, device: str, dtype: str, demo: bool):
    """Return ``(fresh_adapter_fn, tokenizer)``.

    Each condition needs its *own* model instance, because (B) mutates weights
    and (B)–(D) append rows.  Sharing one would leak one condition's changes
    into the next, which is the classic way a benchmark quietly stops measuring
    what it claims to.

    Claim: infrastructure — isolation between conditions is a correctness
    property of the benchmark, not a convenience.
    """
    if demo or model_id == "tiny":
        from parity.tiny import build_tiny_model, build_tiny_tokenizer

        sample = load_embedded_sample()
        tok = build_tiny_tokenizer(sample.by_lang["en"] * 8, vocab_size=900)

        def fresh():
            return TorchLMAdapter(build_tiny_model(tok.vocab_size, seed=0, tie=False), name="tiny")

        return fresh, tok

    from parity.tokenization import HFTokenizer

    tok = HFTokenizer.from_pretrained(model_id)

    def fresh():
        return TorchLMAdapter.from_pretrained(model_id, dtype=dtype, device=device)

    return fresh, tok


def _router(adapter, tokenizer, packs=()):
    aug = AugmentedTokenizer(tokenizer)
    if packs:
        aug = attach_and_verify(adapter, tokenizer, list(packs))
    return MultiTokenizerRouter(adapter, aug)


# ---------------------------------------------------------------------------
# One language
# ---------------------------------------------------------------------------


def run_language(
    fresh_adapter: Callable[[], TorchLMAdapter],
    tokenizer,
    corpus: Corpus,
    parallel: ParallelCorpus,
    cfg: BuildConfig,
    model_id: str,
    train_steps: int = 120,
    max_eval_items: Optional[int] = 48,
    eval_sentences: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all four conditions for one language and return a results dict.

    Claim: reduction, non-regression, bound, low-cost — produces one language's
    worth of evidence for every claim the project makes.
    """
    lang = cfg.lang
    out: Dict[str, Any] = {"lang": lang, "conditions": [], "corpus_source": corpus.source}
    eval_par = parallel
    if eval_sentences:
        eval_par = ParallelCorpus(
            by_lang={k: v[:eval_sentences] for k, v in parallel.by_lang.items()},
            source=parallel.source,
            split=parallel.split,
        )

    # -- (A) status quo -----------------------------------------------------
    log.info("[%s] condition A: original model, original vocabulary", lang)
    adapter_a = fresh_adapter()
    router_a = _router(adapter_a, tokenizer)
    base = evaluate_condition(router_a, eval_par, lang, "base", "A", None, max_eval_items)
    out["conditions"].append(base.to_dict())
    del adapter_a, router_a

    # -- (D) Parity ---------------------------------------------------------
    log.info("[%s] condition D: Parity", lang)
    adapter_d = fresh_adapter()
    t0 = time.time()
    build = build_pack(adapter_d, tokenizer, corpus, cfg, base_model_id=model_id)
    log.info("[%s] built %d tokens in %.1fs", lang, len(build.pack), time.time() - t0)
    if not len(build.pack):
        out["error"] = "no tokens cleared the drift tolerance; nothing to compare"
        return out

    router_d = _router(adapter_d, tokenizer, [build.pack])
    d = evaluate_condition(router_d, eval_par, lang, lang, "D", base, max_eval_items)
    d.n_new_tokens = len(build.pack)
    d.certified_kl_bound = build.pack.worst_bound("kl_next_token")
    d.build_flops = build.total_flops
    d.build_seconds = build.total_seconds
    out["conditions"].append(d.to_dict())
    out["build"] = build.manifest()

    # -- (3) certificate re-verification on fresh contexts ------------------
    splits = split_corpus(corpus)
    fresh_docs = [tokenizer.encode(l) for l in splits.eval.lines]
    shipped = {e.candidate.key for e in build.pack.entries}
    tokens = [t for t in build.synthesized if t.candidate.key in shipped]
    idx = CalibrationIndex([t.candidate for t in tokens], cfg.synthesis.prefix_tokens, cfg.synthesis.suffix_tokens)
    idx.scan(fresh_docs, max_per_candidate=cfg.certify_contexts)
    verdict = verify_certificates(
        DriftCertifier(adapter_d, cfg.certifier),
        tokens,
        {e.candidate.key: e.certificate for e in build.pack.entries},
        idx,
    )
    out["certificate_check"] = {
        "lang": lang,
        "n_tokens": len(build.pack),
        "certified_kl": build.pack.worst_bound("kl_next_token"),
        "measured_kl": max((c.max_observed for c in verdict.coverages if c.statistic == "kl_next_token"), default=0.0),
        "mean_coverage": verdict.mean_coverage,
        "mean_target": verdict.mean_target,
        "violation_rate": verdict.violation_rate,
        "violation_allowance": verdict.violation_allowance,
        "inside": verdict.ok,
        "acceptance_rate": build.acceptance_rate,
    }

    # -- (6) serving throughput --------------------------------------------
    reqs: List[Request] = []
    for i, s in enumerate(eval_par.by_lang[lang][:16]):
        reqs.append(Request(s, lang, 8, f"{lang}-{i}"))
    for i, s in enumerate(eval_par.by_lang["en"][:16]):
        reqs.append(Request(s, "base", 8, f"en-{i}"))
    out["serving"] = router_d.compare_single_vs_multi(reqs)
    del router_d

    # -- (C) zero-shot transfer, same tokens --------------------------------
    log.info("[%s] condition C: zero-shot tokenizer transfer", lang)
    adapter_c = fresh_adapter()
    cands = [e.candidate for e in build.pack.entries]
    fit_docs = [tokenizer.encode(l) for l in splits.fit.lines]
    fit_idx = CalibrationIndex(cands, cfg.synthesis.prefix_tokens, cfg.synthesis.suffix_tokens)
    fit_idx.scan(fit_docs, max_per_candidate=cfg.fit_contexts)
    c_tokens = EmbeddingSynthesizer(adapter_c, zero_shot_transfer_config()).synthesize(cands, fit_idx)
    aug_c = AugmentedTokenizer(tokenizer)
    pack_c = pack_from_tokens(c_tokens, lang, model_id, aug_c)
    router_c = _router(adapter_c, tokenizer, [pack_c])
    c = evaluate_condition(router_c, eval_par, lang, lang, "C", base, max_eval_items)
    c.n_new_tokens = len(pack_c)
    c.build_flops = 0.0  # composition only: no forward passes at all
    out["conditions"].append(c.to_dict())
    del adapter_c, router_c

    # -- (B) vocabulary expansion + continued pretraining -------------------
    log.info("[%s] condition B: continued pretraining (%d steps)", lang, train_steps)
    adapter_b = fresh_adapter()
    b_tokens = EmbeddingSynthesizer(adapter_b, zero_shot_transfer_config()).synthesize(cands, fit_idx)
    aug_b_tmp = AugmentedTokenizer(tokenizer)
    pack_b = pack_from_tokens(b_tokens, lang, model_id, aug_b_tmp)
    aug_b = attach_and_verify(adapter_b, tokenizer, [pack_b])
    training = continued_pretraining(
        adapter_b,
        aug_b,
        aug_b.view(lang),
        splits.mine.lines + splits.fit.lines,
        TrainingConfig(steps=train_steps, batch_sentences=8, lr=1e-4),
    )
    router_b = MultiTokenizerRouter(adapter_b, aug_b)
    b = evaluate_condition(router_b, eval_par, lang, lang, "B", base, max_eval_items)
    b.n_new_tokens = len(pack_b)
    b.build_flops = training.flops
    b.build_seconds = training.seconds
    out["conditions"].append(b.to_dict())
    out["training"] = training.to_dict()
    del router_b

    # -- (4) cost -----------------------------------------------------------
    out["cost"] = costmod.compare_costs(
        build,
        training,
        n_params=adapter_b.n_params,
        added_params=adapter_d.embedding_param_cost(len(build.pack)),
    ).to_dict()
    del adapter_b, adapter_d
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark sweep and write ``results.json``, ``README.md`` and the figure.

    Claim: reduction, non-regression, bound, low-cost — the harness that turns
    the implementation into evidence.
    """
    p = argparse.ArgumentParser(description="Parity four-condition benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--langs", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--split", default="dev")
    p.add_argument("--corpus-lines", type=int, default=997)
    p.add_argument("--eval-sentences", type=int, default=None)
    p.add_argument("--max-eval-items", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--max-kl", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--demo", action="store_true", help="offline fixture model + corpus; illustrative only")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.offline or args.demo:
        import os

        os.environ["PARITY_OFFLINE"] = "1"

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    fresh, tokenizer = make_factory(args.model, args.device, args.dtype, args.demo)

    if args.demo:
        sample = load_embedded_sample()
        parallel = sample
        corpora = {l: Corpus(l, expand_for_testing(sample.by_lang[l], 552, seed=1), source="fixture:expanded_sample") for l in langs}
        certifier = CertifierConfig(
            max_kl=2.0, max_tv=0.6, max_emit_logprob_err=40.0, max_offcontext_mass=0.05, alpha=0.2, delta=0.2, min_calibration=8
        )
        synth = SynthesisConfig(subspace_dim=6, gn_iters=2, max_contexts=6, chunk_size=16, output_contexts=24)
        miner = MinerConfig(min_count=3, min_doc_count=2, max_length=6)
        certify_contexts = 32
        budget = min(args.budget, 48)
    else:
        parallel = load_parallel(langs, split=args.split, max_sentences=args.corpus_lines, allow_download=not args.offline)
        corpora = {l: parallel.monolingual(l) for l in langs}
        certifier = CertifierConfig(max_kl=args.max_kl, alpha=args.alpha, delta=args.delta)
        synth = SynthesisConfig()
        miner = MinerConfig()
        certify_contexts = 64
        budget = args.budget

    results: List[Dict[str, Any]] = []
    for lang in langs:
        cfg = BuildConfig(
            lang=lang,
            budget=budget,
            miner=miner,
            synthesis=synth,
            certifier=certifier,
            selection=SelectionConfig(budget=budget),
            certify_contexts=certify_contexts,
        )
        try:
            results.append(
                run_language(
                    fresh,
                    tokenizer,
                    corpora[lang],
                    parallel,
                    cfg,
                    args.model,
                    train_steps=args.train_steps,
                    max_eval_items=args.max_eval_items,
                    eval_sentences=args.eval_sentences,
                )
            )
        except Exception as exc:  # keep the sweep alive; record the failure
            log.exception("language %s failed", lang)
            results.append({"lang": lang, "error": repr(exc)})

    out_dir = Path(args.out)
    meta = {
        "model": args.model if not args.demo else "tiny (offline fixture)",
        "corpus_source": parallel.source,
        "n_sentences": parallel.n,
        "langs": langs,
        "budget": budget,
        "max_kl": certifier.max_kl,
        "alpha": certifier.alpha,
        "delta": certifier.delta,
        "parity_version": _version.__version__,
        "demo": args.demo,
    }
    payload = {"meta": meta, "languages": results}
    reportmod.write_json(out_dir / "results.json", payload)

    rows = [c for r in results for c in r.get("conditions", [])]
    cost_rows = [r["cost"] for r in results if "cost" in r]
    sections = {"Results (metrics 1, 2, 5)": reportmod.results_table(rows)}
    checks = [r["certificate_check"] for r in results if "certificate_check" in r]
    if checks:
        sections["Certificate re-verification (metric 3)"] = reportmod.certificate_table(checks)
    if cost_rows:
        sections["Cost (metric 4)"] = reportmod.cost_table(cost_rows[0])
    serving = next((r["serving"] for r in results if "serving" in r), None)
    if serving:
        sections["Multi-tokenizer serving (metric 6)"] = reportmod.serving_table(serving)
    reportmod.write_report(out_dir / "README.md", sections, meta)

    try:
        from figures.make_pareto import render

        render(payload, out_dir / "pareto.png")
        sections_note = f", figure {out_dir / 'pareto.png'}"
    except Exception as exc:  # matplotlib optional
        log.warning("figure not rendered: %s", exc)
        sections_note = ""

    print(f"\nwrote {out_dir/'results.json'}, {out_dir/'README.md'}{sections_note}")
    for r in results:
        if "error" in r:
            print(f"  {r['lang']}: ERROR {r['error']}")
            continue
        d = next((c for c in r["conditions"] if c["condition"] == "D"), None)
        if d:
            print(
                f"  {r['lang']}: -{100 * d['token_reduction']:.1f}% tokens, "
                f"Δbits/char {d['bpc_delta']:+.4f}, English Δ {d['english_bpc_delta']:+.2e}, "
                f"certified KL <= {d['certified_kl_bound']:.4g}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
