"""Publish the repository to GitHub and the three Hugging Face artefacts.

    python scripts/publish.py all      --owner NagaYu --dry-run
    python scripts/publish.py github   --owner NagaYu
    python scripts/publish.py space    --owner NagaYu
    python scripts/publish.py dataset  --owner NagaYu
    python scripts/publish.py model    --owner NagaYu --pack packs/ja-smollm2-135m

Why this is a script rather than a runbook: the Space card and the model cards
must not drift from the repository they describe. The Space card is *generated*
from `README.md` plus a frontmatter block defined here, and the model card is
generated from the pack manifest by `scripts/push_vocab_pack.py`. Neither is
hand-maintained, so neither can quietly disagree with the evidence.

`--dry-run` prints exactly what would be created and writes the generated cards
to `build/` without contacting anything.
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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("parity.publish")

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

GITHUB_DESCRIPTION = (
    "Certified vocabulary augmentation without continued pretraining: cut non-English "
    "token cost on open-weight models with a proven bound on behaviour drift."
)
GITHUB_TOPICS = [
    "tokenizer",
    "multilingual",
    "llm",
    "inference-optimization",
    "vocabulary-adaptation",
    "conformal-prediction",
    "submodular-optimization",
    "nlp",
]

#: The published Space is **static**: it tokenizes in the visitor's browser with
#: transformers.js and reads a pack's manifest straight from its Model repo.
#:
#: Two reasons, in order of importance. Hugging Face only hosts Gradio and Docker
#: Spaces on paid hardware, and a demo that costs money to keep up is a demo that
#: eventually goes down. And a static page has no cold start, so the first thing a
#: visitor sees is their own text tokenized, not a queue.
#:
#: ``app.py`` (Gradio) stays in the repository and is the reference
#: implementation — run it locally, or deploy it if you have paid Space hardware.
SPACE_FILES = ["LICENSE"]
SPACE_DIR = "space"

SPACE_FRONTMATTER = """---
title: Parity
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
language:
- ja
- hi
- ar
- th
- sw
- en
tags:
- tokenizer
- vocabulary-adaptation
- multilingual
- inference-efficiency
- certified
short_description: Cut non-English token cost, with a certified bound
---
"""


def run(cmd: Sequence[str], cwd: Optional[Path] = None, check: bool = True, dry: bool = False) -> str:
    """Run a command, echoing it so a dry run reads like a transcript.

    Claim: infrastructure.
    """
    printable = " ".join(cmd)
    if dry:
        print(f"  $ {printable}")
        return ""
    log.info("$ %s", printable)
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {printable}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


def space_card(owner: str, pack_repos: Sequence[str] = ()) -> str:
    """Generate the Space card from the GitHub README plus Space frontmatter.

    One source of truth: the prose lives in ``README.md``, and this only prepends
    the metadata block Spaces need and rewrites relative links to absolute ones,
    since the Space repo does not carry `docs/` links' targets in the same place.

    Claim: infrastructure — a hand-maintained second copy of the README is a
    second place for the claims to be wrong.
    """
    body = (ROOT / "README.md").read_text(encoding="utf-8")
    base = f"https://github.com/{owner}/parity/blob/main/"
    for rel in ("docs/framing.md", "docs/contributing-a-pack.md", "LICENSE", "serving/", "figures/pareto.png"):
        body = body.replace(f"]({rel})", f"]({base}{rel})")
    if pack_repos:
        note = (
            "\n> **Live packs in this Space:** "
            + ", ".join(f"[`{r}`](https://huggingface.co/{r})" for r in pack_repos)
            + "\n"
        )
        body = body.replace("\n---\n\n## The problem", note + "\n---\n\n## The problem", 1)
    return SPACE_FRONTMATTER + "\n" + body


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def publish_github(owner: str, repo: str, private: bool = False, dry: bool = False) -> str:
    """Create (or update) the GitHub repository and push ``main``.

    Claim: infrastructure.
    """
    url = f"https://github.com/{owner}/{repo}"
    if not (ROOT / ".git").exists():
        run(["git", "init", "-b", "main"], dry=dry)
    exists = False
    if not dry:
        try:
            run(["gh", "repo", "view", f"{owner}/{repo}"])
            exists = True
        except RuntimeError:
            exists = False
    if not exists:
        run(
            [
                "gh", "repo", "create", f"{owner}/{repo}",
                "--private" if private else "--public",
                "--description", GITHUB_DESCRIPTION,
                "--source", ".", "--remote", "origin",
            ],
            dry=dry,
        )
    else:
        run(["git", "remote", "remove", "origin"], check=False, dry=dry)
        run(["git", "remote", "add", "origin", f"{url}.git"], dry=dry)

    run(["git", "add", "-A"], dry=dry)
    run(["git", "commit", "-m", "Parity: certified vocabulary augmentation without continued pretraining"],
        check=False, dry=dry)
    run(["git", "branch", "-M", "main"], dry=dry)
    run(["git", "push", "-u", "origin", "main"], dry=dry)
    if GITHUB_TOPICS:
        run(["gh", "repo", "edit", f"{owner}/{repo}", "--add-topic", ",".join(GITHUB_TOPICS)], check=False, dry=dry)
    return url


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


def publish_space(owner: str, name: str, pack_repos: Sequence[str] = (), private: bool = False, dry: bool = False) -> str:
    """Create the Gradio Space and upload only what it needs to run.

    Claim: infrastructure — the demo that makes the reduction visible to someone
    who will never clone the repository.
    """
    card = space_card(owner, pack_repos)
    staging = BUILD / "space"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "README.md").write_text(card, encoding="utf-8")
    for item in SPACE_FILES:
        src = ROOT / item
        if src.exists():
            shutil.copy2(src, staging / item)
    for src in sorted((ROOT / SPACE_DIR).iterdir()):
        if src.name.startswith("."):
            continue
        shutil.copy2(src, staging / src.name)

    repo_id = f"{owner}/{name}"
    if dry:
        print(f"  would create static Space {repo_id} and upload {sorted(p.name for p in staging.iterdir())}")
        return f"https://huggingface.co/spaces/{repo_id}"

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="space")
    return f"https://huggingface.co/spaces/{repo_id}"


def publish_dataset(owner: str, name: str, atlas: Path, private: bool = False, dry: bool = False) -> str:
    """Push the fertility atlas as a Dataset.

    Claim: reduction — makes the baseline every claim is relative to auditable
    by someone who did not run the code.
    """
    if not atlas.exists():
        raise FileNotFoundError(f"{atlas} not found — run scripts/build_atlas.py first")
    rows = [json.loads(l) for l in atlas.read_text(encoding="utf-8").splitlines() if l.strip()]
    repo_id = f"{owner}/{name}"
    if dry:
        print(f"  would push {len(rows)} atlas rows to dataset {repo_id}")
        return f"https://huggingface.co/datasets/{repo_id}"

    from scripts.build_atlas import DATASET_CARD, push

    card = DATASET_CARD.format(
        corpus=rows[0].get("corpus", "OPUS-100"),
        tokenizers=",".join(sorted({r["tokenizer_id"] for r in rows})),
        langs=",".join(sorted({r["lang"] for r in rows})),
        n=rows[0].get("n_sentences", "?"),
    )
    return push(rows, repo_id, card, private)


def publish_model(owner: str, name: str, pack_dir: Path, private: bool = False, dry: bool = False) -> str:
    """Push a vocabulary pack as a Model repo, card generated from the manifest.

    Claim: bound — the published artefact carries its own certificate, and the
    card states it because the card is derived from it.
    """
    from parity.pack import load_pack

    from scripts.push_vocab_pack import build_card, summarize, validate

    pack = load_pack(pack_dir, require_accepted=False)
    problems = validate(pack)
    summary = summarize(pack)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if problems:
        raise RuntimeError("pack is not fit to publish: " + "; ".join(problems))

    repo_id = f"{owner}/{name}"
    card = build_card(pack, repo_id, allow_uncertified=False)
    BUILD.mkdir(exist_ok=True)
    (BUILD / f"{name}.README.md").write_text(card, encoding="utf-8")
    if dry:
        print(f"  would push {len(pack)} tokens to model {repo_id}")
        return f"https://huggingface.co/{repo_id}"

    from scripts.push_vocab_pack import push

    return push(pack_dir, repo_id, card, private)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Publish Parity to GitHub and Hugging Face")
    ap.add_argument("what", choices=["all", "github", "space", "dataset", "model"])
    ap.add_argument("--owner", required=True, help="GitHub user/org and HF user/org (assumed the same)")
    ap.add_argument("--repo", default="parity")
    ap.add_argument("--space", default="parity")
    ap.add_argument("--dataset", default="parity-fertility-atlas")
    ap.add_argument("--model-name", default=None, help="model repo name; defaults to the pack directory name")
    ap.add_argument("--pack", default=None, help="pack directory to publish as a Model")
    ap.add_argument("--atlas", default="data/atlas.jsonl")
    ap.add_argument("--pack-repos", default="", help="comma-separated pack repo ids for the Space to load")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    sys.path.insert(0, str(ROOT))
    pack_repos = [r.strip() for r in args.pack_repos.split(",") if r.strip()]
    urls: Dict[str, str] = {}
    dry = args.dry_run

    if args.what in ("all", "github"):
        print("GitHub:")
        urls["github"] = publish_github(args.owner, args.repo, args.private, dry)
    if args.what in ("all", "model") and args.pack:
        print("Model:")
        pack_dir = Path(args.pack)
        name = args.model_name or f"parity-{pack_dir.name}"
        urls["model"] = publish_model(args.owner, name, pack_dir, args.private, dry)
        pack_repos = sorted(set(pack_repos) | {f"{args.owner}/{name}"})
    if args.what in ("all", "dataset"):
        print("Dataset:")
        try:
            urls["dataset"] = publish_dataset(args.owner, args.dataset, Path(args.atlas), args.private, dry)
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
    if args.what in ("all", "space"):
        print("Space:")
        urls["space"] = publish_space(args.owner, args.space, pack_repos, args.private, dry)

    print("\n" + ("(dry run) " if dry else "") + "published:")
    for k, v in urls.items():
        print(f"  {k:<8} {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
