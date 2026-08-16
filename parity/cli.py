"""``parity`` command line.

    parity build --model Qwen/Qwen2.5-0.5B-Instruct --lang ja --budget 8000
    parity atlas --model Qwen/Qwen2.5-0.5B-Instruct --langs ja,hi,ar,th,sw
    parity verify --pack packs/ja --model Qwen/Qwen2.5-0.5B-Instruct
    parity inspect --pack packs/ja
    parity demo --lang ja

``demo`` runs the entire pipeline on a small offline fixture model in a couple
of seconds, which is the fastest way to see the shape of the output before
committing a GPU to a real build.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

log = logging.getLogger("parity.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_model(args):
    """Load either a real open-weight model or the offline fixture.

    Claim: infrastructure — one code path for demos and production builds, so
    the demo cannot drift from the thing it demonstrates.
    """
    from parity.adapters import TorchLMAdapter

    if args.model == "tiny":
        from parity.corpora import load_embedded_sample
        from parity.tiny import build_tiny_model, build_tiny_tokenizer

        sample = load_embedded_sample()
        # English-only training corpus: a fixture tokenizer that had seen the
        # target sentences would memorise them and make the demo meaningless.
        tok = build_tiny_tokenizer(sample.by_lang["en"] * 8, vocab_size=900)
        model = build_tiny_model(tok.vocab_size, seed=0, tie=getattr(args, "tie", False))
        return TorchLMAdapter(model, name="tiny"), tok
    from parity.tokenization import HFTokenizer

    adapter = TorchLMAdapter.from_pretrained(args.model, dtype=args.dtype, device=args.device)
    return adapter, HFTokenizer.from_pretrained(args.model)


def _load_corpus(args, lang: str):
    """Load the mining/calibration corpus for one language.

    Claim: infrastructure.
    """
    from parity.corpora import Corpus, expand_for_testing, load_embedded_sample, load_monolingual

    if args.model == "tiny" or args.fixture_corpus:
        sample = load_embedded_sample()
        lines = expand_for_testing(sample.by_lang[lang], args.corpus_lines, seed=1)
        return Corpus(lang, lines, source="fixture:expanded_sample")
    return load_monolingual(lang, split=args.split, max_sentences=args.corpus_lines, allow_download=not args.offline)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_build(args) -> int:
    """Mine, synthesise, certify, select and write a vocabulary pack.

    Claim: reduction, non-regression, bound, low-cost — the command that
    produces the artefact all four claims are about.
    """
    from parity.build import BuildConfig, build_pack
    from parity.certificate import CertifierConfig
    from parity.miner import MinerConfig
    from parity.pack import save_pack
    from parity.selection import SelectionConfig
    from parity.synthesis import SynthesisConfig

    adapter, tokenizer = _load_model(args)
    corpus = _load_corpus(args, args.lang)
    if corpus.source.startswith("fixture") and not args.allow_fixture:
        log.warning("building from the offline fixture corpus; numbers are illustrative, not evidence")

    cfg = BuildConfig(
        lang=args.lang,
        budget=args.budget,
        oversample=args.oversample,
        miner=MinerConfig(min_count=args.min_count, max_length=args.max_merge_len),
        synthesis=SynthesisConfig(
            solver=args.solver,
            subspace_dim=args.subspace_dim,
            gn_iters=args.gn_iters,
            max_contexts=args.contexts,
        ),
        certifier=CertifierConfig(
            max_kl=args.max_kl,
            max_tv=args.max_tv,
            max_emit_logprob_err=args.max_emit_err,
            max_offcontext_mass=args.max_offcontext_mass,
            alpha=args.alpha,
            delta=args.delta,
        ),
        selection=SelectionConfig(budget=args.budget),
        certify_contexts=args.certify_contexts,
        seed=args.seed,
    )
    result = build_pack(adapter, tokenizer, corpus, cfg, base_model_id=args.model)

    out = Path(args.out or f"packs/{args.lang}")
    save_pack(result.pack, out)
    (out / "build_manifest.json").write_text(
        json.dumps(result.manifest(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    m = result.manifest()
    print(f"\npack: {out}")
    print(f"  language              {args.lang}")
    print(f"  tokens adopted        {len(result.pack)} / budget {args.budget}")
    print(f"  token reduction       {100 * result.selection.token_reduction:.1f}%  (held-out eval slice)")
    print(f"  effective context     x{1 / max(1e-9, 1 - result.selection.token_reduction):.2f}")
    print(f"  certified KL bound    {m['worst_kl_bound']:.4g} nats (worst token, {100 * (1 - cfg.certifier.alpha):.0f}%/{100 * (1 - cfg.certifier.delta):.0f}%)")
    print(f"  certified TV bound    {m['worst_tv_bound']:.4g}")
    print(f"  acceptance rate       {100 * result.acceptance_rate:.1f}% of certified candidates")
    print(f"  residual reduction    {100 * result.mean_residual_reduction:.1f}% vs composition baseline")
    print(f"  optimality (certified) >= {result.selection.certified_optimality_ratio:.3f} of the best pack this size")
    print(f"  build cost            {result.total_flops:.3e} FLOPs, {result.total_seconds:.1f}s")
    print(f"  English               unchanged by construction (append-only rows + base-view mask)")
    rejected = result.rejection_reasons()
    if rejected:
        print("  candidates refused:")
        for reason, n in rejected:
            print(f"    {n:>6}  {reason}")
        if any("occurrences" in r or "measurements" in r for r, _ in rejected):
            print(
                "    (refusals for evidence, not drift: enlarge the calibration corpus via\n"
                f"     PARITY_MINING_CORPUS_{args.lang.upper()}, or relax --alpha/--delta)"
            )
    return 0


def cmd_atlas(args) -> int:
    """Print (and optionally write) a language x tokenizer fertility table.

    Claim: reduction — the baseline measurement the whole project is scored
    against.
    """
    from parity.corpora import DEFAULT_TARGETS, load_parallel
    from parity.fertility import fertility_table
    from parity.tokenization import AugmentedTokenizer

    langs = [l.strip() for l in (args.langs or ",".join(DEFAULT_TARGETS)).split(",") if l.strip()]
    adapter_tok = None
    if args.model == "tiny":
        from parity.corpora import load_embedded_sample
        from parity.tiny import build_tiny_tokenizer

        sample = load_embedded_sample()
        adapter_tok = build_tiny_tokenizer(sample.by_lang["en"] * 8, vocab_size=900)
    else:
        from parity.tokenization import HFTokenizer

        adapter_tok = HFTokenizer.from_pretrained(args.model)

    corpus = load_parallel(langs, split=args.split, max_sentences=args.corpus_lines, allow_download=not args.offline)
    aug = AugmentedTokenizer(adapter_tok)
    table = fertility_table(aug, corpus, langs + ["en"], tokenizer_id=args.model)

    print(f"\ncorpus: {corpus.source} ({corpus.n} aligned sentences)\n")
    print(f"{'lang':<6}{'tokens/char':>13}{'tokens/word':>13}{'vs English':>12}{'median':>10}{'eff. context':>14}")
    print("-" * 68)
    for lang, rep in sorted(table.items(), key=lambda kv: -(kv[1].parity_ratio or 0)):
        tpw = rep.tokens_per_word
        med = rep.parity_ratio_median
        print(
            f"{lang:<6}{rep.tokens_per_char:>13.3f}"
            f"{(f'{tpw:.3f}' if tpw else '—'):>13}"
            f"{(f'{rep.parity_ratio:.2f}x' if rep.parity_ratio else '1.00x'):>12}"
            f"{(f'{med:.2f}x' if med else '—'):>10}"
            f"{(f'{100 * rep.effective_context_fraction:.0f}%' if rep.effective_context_fraction else '100%'):>14}"
        )
    if corpus.source == "opus100":
        print(
            "\nCorpus-level ('vs English') and per-sentence median ratios are both shown because\n"
            "OPUS-100 is web-mined: a large gap between them means alignment noise survived cleaning.\n"
            "FLORES-200 (gated; accept terms + `huggingface-cli login`) needs no such caveat."
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps({k: v.to_dict() for k, v in table.items()}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_verify(args) -> int:
    """Re-measure drift on fresh contexts and check it against the certificate.

    Claim: bound — makes the guarantee falsifiable from the command line, by
    someone who did not build the pack.
    """
    from parity.certificate import CertifierConfig, DriftCertifier, verify_certificates
    from parity.pack import load_pack
    from parity.synthesis import CalibrationIndex
    from parity.types import SynthesizedToken

    adapter, tokenizer = _load_model(args)
    pack = load_pack(args.pack)
    corpus = _load_corpus(args, pack.lang)
    docs = [tokenizer.encode(l) for l in corpus.lines]

    tokens = [
        SynthesizedToken(
            candidate=e.candidate,
            input_embedding=pack.input_embeddings[i],
            output_embedding=None if pack.output_embeddings is None else pack.output_embeddings[i],
            solver=e.solver,
        )
        for i, e in enumerate(pack.entries)
    ]
    index = CalibrationIndex([t.candidate for t in tokens], 12, 6)
    index.scan(docs, max_per_candidate=args.contexts)
    certifier = DriftCertifier(adapter, CertifierConfig(alpha=args.alpha, delta=args.delta))
    certs = {e.candidate.key: e.certificate for e in pack.entries}
    result = verify_certificates(certifier, tokens, certs, index)

    print(f"\nverified pack {args.pack} ({len(pack)} tokens)")
    print(f"  checks run       {result.n_checked}")
    print(f"  mean coverage    {100 * result.mean_coverage:.1f}% (target {100 * result.mean_target:.0f}%)")
    print(f"  under-covering   {len(result.violations)} of {result.n_checked} bounds")
    print(
        f"  rate             {100 * result.violation_rate:.2f}% "
        f"(guarantee permits up to {100 * result.allowed_violation_rate:.0f}%, "
        f"test threshold {100 * result.violation_allowance:.1f}%)"
    )
    for key, stat, got, target in result.violations[:10]:
        print(f"    {key} {stat}: coverage {100 * got:.1f}% < target {100 * target:.0f}%")
    print("  RESULT:", "PASS" if result.ok else "FAIL")
    return 0 if result.ok else 1


def cmd_inspect(args) -> int:
    """Print a pack's contents: tokens, savings and certified bounds.

    Claim: bound — the tokens are printed as strings so that a speaker of the
    language can review what was added without reading any code.
    """
    from parity.pack import load_pack

    pack = load_pack(args.pack, require_accepted=False)
    print(f"\npack {args.pack}")
    print(f"  language        {pack.lang}")
    print(f"  base model      {pack.base_model_id}")
    print(f"  base vocab      {pack.base_vocab_size}")
    print(f"  tokens          {len(pack)}")
    print(f"  worst KL bound  {pack.worst_bound('kl_next_token'):.4g}")
    print(f"  worst TV bound  {pack.worst_bound('tv_next_token'):.4g}\n")
    print(f"{'#':>5}  {'surface':<24}{'len':>5}{'count':>8}{'KL bound':>12}")
    print("-" * 60)
    for i, e in enumerate(pack.entries[: args.limit]):
        spec = e.certificate.bounds.get("kl_next_token")
        surface = e.candidate.surface.replace("\n", "\\n")
        print(
            f"{i:>5}  {surface[:24]:<24}{e.candidate.length:>5}{e.candidate.count:>8}"
            f"{(f'{spec.value:.4g}' if spec else '—'):>12}"
        )
    if len(pack) > args.limit:
        print(f"... {len(pack) - args.limit} more (use --limit)")
    return 0


def cmd_demo(args) -> int:
    """Run the whole pipeline offline on the fixture model, in seconds.

    Claim: infrastructure — the fastest possible way for a reviewer to see what
    the pipeline produces before spending anything.
    """
    args.model = "tiny"
    args.fixture_corpus = True
    args.allow_fixture = True
    rc = cmd_build(args)
    print("\n(demo used the offline fixture model and corpus: shapes are real, magnitudes are not)")
    return rc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Claim: infrastructure.
    """
    p = argparse.ArgumentParser(prog="parity", description="Certified vocabulary augmentation without continued pretraining.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hub id, local path, or 'tiny' for the offline fixture")
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
        sp.add_argument("--split", default="dev", help="FLORES split")
        sp.add_argument("--corpus-lines", type=int, default=997)
        sp.add_argument("--offline", action="store_true", help="never touch the network")
        sp.add_argument("--fixture-corpus", action="store_true", help="use the embedded sample instead of FLORES")
        sp.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("build", help="build a certified vocabulary pack")
    common(b)
    b.add_argument("--lang", required=True)
    b.add_argument("--budget", type=int, default=8000, help="number of new embedding rows")
    b.add_argument("--out", default=None)
    b.add_argument("--oversample", type=float, default=2.5, help="certify this multiple of the budget")
    b.add_argument("--min-count", type=int, default=3)
    b.add_argument("--max-merge-len", type=int, default=8)
    b.add_argument("--solver", default="gn", choices=["composition", "gn", "gn+adam"])
    b.add_argument("--subspace-dim", type=int, default=8)
    b.add_argument("--gn-iters", type=int, default=2)
    b.add_argument("--contexts", type=int, default=8)
    b.add_argument("--max-kl", type=float, default=0.05, help="drift tolerance, nats")
    b.add_argument("--max-tv", type=float, default=0.05)
    b.add_argument("--max-emit-err", type=float, default=0.75, help="emission log-prob tolerance, nats")
    b.add_argument("--max-offcontext-mass", type=float, default=0.01, help="max probability the token may take off-context")
    b.add_argument("--certify-contexts", type=int, default=64, help="held-out occurrences per candidate")
    b.add_argument("--alpha", type=float, default=0.05, help="tail fraction the bound must cover")
    b.add_argument("--delta", type=float, default=0.05, help="confidence in the bound")
    b.add_argument("--allow-fixture", action="store_true")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("atlas", help="fertility of each language under this tokenizer")
    common(a)
    a.add_argument("--langs", default=None)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_atlas, lang="ja")

    v = sub.add_parser("verify", help="re-check a pack's certificates on fresh data")
    common(v)
    v.add_argument("--pack", required=True)
    v.add_argument("--contexts", type=int, default=24)
    v.add_argument("--alpha", type=float, default=0.05)
    v.add_argument("--delta", type=float, default=0.05)
    v.set_defaults(func=cmd_verify, lang="ja")

    i = sub.add_parser("inspect", help="print a pack's tokens and bounds")
    i.add_argument("--pack", required=True)
    i.add_argument("--limit", type=int, default=40)
    i.set_defaults(func=cmd_inspect)

    d = sub.add_parser("demo", help="run the full pipeline offline in seconds")
    common(d)
    d.add_argument("--lang", default="ja")
    d.add_argument("--budget", type=int, default=64)
    d.add_argument("--out", default="packs/demo-ja")
    d.add_argument("--oversample", type=float, default=2.0)
    d.add_argument("--min-count", type=int, default=3)
    d.add_argument("--max-merge-len", type=int, default=6)
    d.add_argument("--solver", default="gn", choices=["composition", "gn", "gn+adam"])
    d.add_argument("--subspace-dim", type=int, default=6)
    d.add_argument("--gn-iters", type=int, default=2)
    d.add_argument("--contexts", type=int, default=6)
    # The fixture model is untrained, so its residual-stream geometry is far
    # noisier than a real checkpoint's. The demo therefore runs at a deliberately
    # loose tolerance: it demonstrates the mechanism, not the magnitudes.
    d.add_argument("--max-kl", type=float, default=2.0)
    d.add_argument("--max-tv", type=float, default=0.6)
    d.add_argument("--max-emit-err", type=float, default=40.0)
    d.add_argument("--max-offcontext-mass", type=float, default=0.05)
    d.add_argument("--certify-contexts", type=int, default=32)
    d.add_argument("--alpha", type=float, default=0.2)
    d.add_argument("--delta", type=float, default=0.2)
    d.set_defaults(func=cmd_demo, corpus_lines=400)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``parity`` console script.

    Claim: infrastructure.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(args.verbose)
    if getattr(args, "offline", False):
        import os

        os.environ["PARITY_OFFLINE"] = "1"
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
