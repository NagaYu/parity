"""Regenerate the static Space's data files from the repository's own sources.

    python scripts/build_space_assets.py

The Space ships two JSON blobs it cannot compute in the browser: the aligned
sample sentences (so the English-equivalent length can be *exact* for those) and
the fertility atlas (so the cross-language table is a measurement, not a
hardcoded claim). Both are derived here rather than hand-edited, and
``tests/test_space.py`` fails if they drift from their sources — a demo quoting
numbers the repository no longer produces is the same failure mode as a model
card quoting a bound the manifest no longer contains.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
SAMPLE_SRC = ROOT / "data" / "parallel_sample.json"
ATLAS_SRC = ROOT / "data" / "atlas.jsonl"


def sample_payload() -> dict:
    """The aligned sample sentences, trimmed to what the page uses.

    Claim: infrastructure.
    """
    return {"sentences": json.loads(SAMPLE_SRC.read_text(encoding="utf-8"))["sentences"]}


def atlas_payload() -> list:
    """The fertility atlas rows, or an empty list if it has not been built.

    Claim: reduction — the Space's cross-language table is the atlas, so it
    cannot say anything the atlas does not.
    """
    if not ATLAS_SRC.exists():
        return []
    return [json.loads(line) for line in ATLAS_SRC.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(space_dir: Path = SPACE) -> list[Path]:
    """Write both payloads into the Space directory.

    Claim: infrastructure.
    """
    space_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, payload in (("sample.json", sample_payload()), ("atlas.json", atlas_payload())):
        p = space_dir / name
        p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        out.append(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Claim: infrastructure.
    """
    ap = argparse.ArgumentParser(description="Regenerate the static Space's data files")
    ap.add_argument("--space-dir", default=str(SPACE))
    args = ap.parse_args(list(argv) if argv is not None else None)
    for p in write(Path(args.space_dir)):
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
