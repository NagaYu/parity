"""Push a vocabulary pack to the Hub as a Model repo, with its certificate.

    python scripts/push_vocab_pack.py --pack packs/ja --repo your-org/parity-ja-qwen2.5-0.5b
    python scripts/push_vocab_pack.py --pack packs/ja --repo ... --dry-run

The one rule this script enforces, and refuses to be talked out of: **the model
card must state the certified drift, and the card is generated from the
manifest.** There is no code path that publishes a pack whose card was written
by hand, and `--dry-run` prints exactly what would be uploaded so the numbers
can be checked before anything leaves the machine.

A pack containing any entry without an accepted certificate is refused outright.
If you want to publish an uncertified pack — a baseline for a paper, say — mark
it with `--allow-uncertified`, which stamps the card with a warning banner that
cannot be removed by editing this script's inputs.
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

log = logging.getLogger("parity.push")

UNCERTIFIED_BANNER = """
> ## ⚠️ This pack is **not certified**
>
> One or more tokens in this pack ship without a drift certificate that cleared
> a tolerance. It is published for comparison or research only. Do not deploy it
> in front of users. The whole point of Parity is the bound; a pack without one
> is a vocabulary transfer baseline, not a Parity pack.
"""


def validate(pack, allow_uncertified: bool = False) -> List[str]:
    """Check a pack is fit to publish; return a list of problems.

    Claim: bound — the gate between "we measured drift" and "someone else runs
    this in front of users".
    """
    problems: List[str] = []
    if len(pack) == 0:
        problems.append("pack contains no tokens")
    if pack.input_embeddings is None:
        problems.append("pack has no embedding rows")
    if not pack.base_tokenizer_fingerprint:
        problems.append("pack has no tokenizer fingerprint; it cannot be safely attached to any model")
    missing = [e.candidate.key for e in pack.entries if not e.certificate.accepted]
    if missing and not allow_uncertified:
        problems.append(f"{len(missing)} tokens have no accepted certificate (use --allow-uncertified to publish anyway)")
    no_bound = [e.candidate.key for e in pack.entries if "kl_next_token" not in e.certificate.bounds]
    if no_bound and not allow_uncertified:
        problems.append(f"{len(no_bound)} tokens have no KL bound at all")
    return problems


def summarize(pack) -> Dict[str, Any]:
    """Numbers a reviewer should see before a push.

    Claim: bound, reduction.
    """
    return {
        "lang": pack.lang,
        "base_model": pack.base_model_id,
        "n_tokens": len(pack),
        "worst_kl_bound": pack.worst_bound("kl_next_token"),
        "worst_tv_bound": pack.worst_bound("tv_next_token"),
        "worst_offcontext_mass": pack.worst_bound("offcontext_mass"),
        "token_reduction": pack.metadata.get("token_reduction"),
        "added_parameters": pack.metadata.get("added_parameters"),
        "build_flops": pack.metadata.get("build_flops"),
        "uncertified_entries": sum(1 for e in pack.entries if not e.certificate.accepted),
    }


def build_card(pack, repo_id: str, allow_uncertified: bool) -> str:
    """Generate the model card from the manifest, never from prose.

    Claim: bound — a card that can disagree with the evidence is a supply-chain
    problem; generating it removes the possibility.
    """
    from parity.pack import model_card

    card = model_card(pack)
    card = card.replace("<this-repo-id>", repo_id)
    if any(not e.certificate.accepted for e in pack.entries):
        # Insert directly after the frontmatter so it cannot be scrolled past.
        head, _, tail = card.partition("---\n\n")
        card = head + "---\n\n" + UNCERTIFIED_BANNER + "\n" + tail
    return card


def push(pack_dir: Path, repo_id: str, card: str, private: bool = False) -> str:
    """Upload the pack directory and the generated card.

    Claim: infrastructure.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(pack_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["README.md"],  # replaced by the generated card below
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Claim: bound, infrastructure.
    """
    ap = argparse.ArgumentParser(description="Publish a Parity vocabulary pack")
    ap.add_argument("--pack", required=True, help="pack directory produced by `parity build`")
    ap.add_argument("--repo", default=None, help="HF model repo id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--allow-uncertified", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the card and summary, upload nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    from parity.pack import load_pack

    pack_dir = Path(args.pack)
    pack = load_pack(pack_dir, require_accepted=False)

    problems = validate(pack, args.allow_uncertified)
    summary = summarize(pack)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if problems:
        print("\nrefusing to publish:")
        for p in problems:
            print(f"  - {p}")
        if not args.dry_run:
            return 1

    repo_id = args.repo or f"parity-{pack.lang}"
    card = build_card(pack, repo_id, args.allow_uncertified)

    if args.dry_run or not args.repo:
        out = pack_dir / "README.generated.md"
        out.write_text(card, encoding="utf-8")
        print(f"\n(dry run) card written to {out}; nothing uploaded")
        print(f"(dry run) would upload {pack_dir} to {repo_id}")
        return 0

    url = push(pack_dir, args.repo, card, args.private)
    print(f"\npushed {len(pack)} tokens to {url}")
    print(f"certified KL <= {summary['worst_kl_bound']:.4g} nats (worst token) — stated on the card")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
