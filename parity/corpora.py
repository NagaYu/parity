"""Parallel and monolingual corpora, with an offline fallback.

Everything Parity claims is measured against a *parallel* corpus, because the
only honest way to compare token cost across languages is to hold meaning
constant.  FLORES-200 (CC BY-SA 4.0) is the default: 200+ languages, the same
sentences in all of them, professionally translated.

If FLORES cannot be downloaded, we fall back to a 24-sentence hand-checked
sample embedded in ``data/parallel_sample.json`` so that tests, the CLI smoke
path and the Space still run.  Every loader records which source it used in
:attr:`Corpus.source`, and :mod:`benchmarks.run` writes that into the run
manifest — a headline number can never silently come from 24 sentences.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("parity.corpora")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_PATH = _REPO_ROOT / "data" / "parallel_sample.json"
_CACHE_DIR = Path(os.environ.get("PARITY_CACHE", _REPO_ROOT / "data" / "flores"))

PIVOT = "en"


@dataclass(frozen=True)
class LanguageSpec:
    """Registry entry for one language.

    ``whitespace_delimited`` decides whether ``tokens_per_word`` is a meaningful
    statistic at all; for Japanese and Thai it is not, which is precisely why
    :attr:`parity.types.FertilityReport.parity_ratio` is the metric Parity
    leads with.

    Claim: reduction — picking the right denominator per script is the
    difference between measuring token cost and measuring an artefact of
    whitespace conventions.
    """

    code: str
    flores: str
    name: str
    endonym: str
    script: str
    whitespace_delimited: bool = True


LANGUAGES: Dict[str, LanguageSpec] = {
    "en": LanguageSpec("en", "eng_Latn", "English", "English", "Latin", True),
    "ja": LanguageSpec("ja", "jpn_Jpan", "Japanese", "日本語", "Japanese", False),
    "hi": LanguageSpec("hi", "hin_Deva", "Hindi", "हिन्दी", "Devanagari", True),
    "ar": LanguageSpec("ar", "arb_Arab", "Arabic", "العربية", "Arabic", True),
    "th": LanguageSpec("th", "tha_Thai", "Thai", "ไทย", "Thai", False),
    "sw": LanguageSpec("sw", "swh_Latn", "Swahili", "Kiswahili", "Latin", True),
    "ko": LanguageSpec("ko", "kor_Hang", "Korean", "한국어", "Hangul", True),
    "bn": LanguageSpec("bn", "ben_Beng", "Bengali", "বাংলা", "Bengali", True),
    "te": LanguageSpec("te", "tel_Telu", "Telugu", "తెలుగు", "Telugu", True),
    "ta": LanguageSpec("ta", "tam_Taml", "Tamil", "தமிழ்", "Tamil", True),
    "am": LanguageSpec("am", "amh_Ethi", "Amharic", "አማርኛ", "Ethiopic", True),
    "my": LanguageSpec("my", "mya_Mymr", "Burmese", "မြန်မာ", "Myanmar", False),
    "zh": LanguageSpec("zh", "zho_Hans", "Chinese (Simplified)", "中文", "Han", False),
    "ru": LanguageSpec("ru", "rus_Cyrl", "Russian", "русский", "Cyrillic", True),
    "es": LanguageSpec("es", "spa_Latn", "Spanish", "español", "Latin", True),
}

#: Languages the default benchmark sweep covers.  Japanese is mandatory per the
#: project brief; the rest are chosen to span four scripts and two
#: word-segmentation regimes.
DEFAULT_TARGETS = ["ja", "hi", "ar", "th", "sw"]


def language(code: str) -> LanguageSpec:
    """Look up a :class:`LanguageSpec`, with a helpful error for typos.

    Claim: infrastructure.
    """
    if code not in LANGUAGES:
        raise KeyError(f"unknown language {code!r}; known: {sorted(LANGUAGES)}")
    return LANGUAGES[code]


@dataclass
class Corpus:
    """A list of lines in one language, plus provenance.

    Claim: infrastructure — provenance travels with the data so that no result
    can be reported without its corpus being identifiable.
    """

    lang: str
    lines: List[str] = field(default_factory=list)
    source: str = "unknown"
    split: str = ""

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def head(self, n: int) -> "Corpus":
        """First ``n`` lines, preserving provenance.

        Claim: infrastructure.
        """
        return Corpus(self.lang, self.lines[:n], self.source, self.split)

    def slice(self, start: int, stop: int) -> "Corpus":
        """Half-open slice, preserving provenance.

        Used to keep mining, calibration and evaluation on disjoint lines —
        a bound fitted on the data it is later checked against is not a bound.

        Claim: bound.
        """
        return Corpus(self.lang, self.lines[start:stop], self.source, self.split)


@dataclass
class ParallelCorpus:
    """Meaning-matched lines, with the pivot each language is aligned against.

    Two alignment regimes, because the available corpora come in both shapes:

    * **Fully aligned** (FLORES-200, the embedded sample): ``by_lang[code][i]``
      means the same thing for every ``code``, and every language shares one
      English pivot.
    * **Pairwise aligned** (OPUS-100): each language comes with *its own* English
      side. Cross-target rows are unrelated, but every target↔English pair is
      valid — which is all the parity ratio needs.

    :meth:`pivot_for` hides the difference, so callers never accidentally compare
    a Hindi sentence against the English translation of a Japanese one.

    Claim: reduction — holding meaning constant is what makes a cross-language
    token-count comparison mean anything, and this is where that is enforced.
    """

    by_lang: Dict[str, List[str]]
    source: str = "unknown"
    split: str = ""
    #: Per-language English side, for pairwise-aligned corpora. Empty for
    #: fully-aligned ones, where ``by_lang[PIVOT]`` serves every language.
    pivot_by_lang: Dict[str, List[str]] = field(default_factory=dict)

    def pivot_for(self, lang: str, pivot: str = PIVOT) -> List[str]:
        """The English lines this language is actually aligned against.

        Claim: reduction — using the wrong pivot silently turns the headline
        metric into noise, so there is exactly one way to ask for it.
        """
        if lang in self.pivot_by_lang:
            return self.pivot_by_lang[lang]
        return self.by_lang.get(pivot, [])

    @property
    def n(self) -> int:
        """Number of aligned sentences.

        Claim: infrastructure.
        """
        return min((len(v) for v in self.by_lang.values()), default=0)

    def langs(self) -> List[str]:
        """Languages present, sorted.

        Claim: infrastructure.
        """
        return sorted(self.by_lang)

    def pair(self, target: str, pivot: str = PIVOT) -> List[tuple]:
        """Zip target and pivot sentences into aligned pairs.

        Claim: reduction — this is the input to the parity-ratio measurement.
        """
        piv = self.pivot_for(target, pivot)
        n = min(len(self.by_lang[target]), len(piv))
        return [(self.by_lang[target][i], piv[i]) for i in range(n)]

    def monolingual(self, lang: str) -> Corpus:
        """Extract one language as a :class:`Corpus`.

        Claim: infrastructure.
        """
        return Corpus(lang, list(self.by_lang[lang]), self.source, self.split)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_embedded_sample(langs: Optional[Sequence[str]] = None) -> ParallelCorpus:
    """Load the 24-sentence offline parallel sample shipped with the repo.

    Claim: infrastructure — guarantees the whole pipeline is runnable with no
    network, which is what makes the test suite a real gate rather than a
    CI-only ritual.
    """
    with open(_SAMPLE_PATH, encoding="utf-8") as fh:
        blob = json.load(fh)
    data = blob["sentences"]
    if langs is not None:
        missing = [l for l in langs if l not in data]
        if missing:
            raise KeyError(f"embedded sample has no {missing}; it covers {sorted(data)}")
        data = {l: data[l] for l in langs}
    return ParallelCorpus(by_lang={k: list(v) for k, v in data.items()}, source="embedded_sample", split="sample")


def _cache_path(flores_code: str, split: str) -> Path:
    return _CACHE_DIR / f"{flores_code}.{split}.txt"


def _load_cached(flores_code: str, split: str) -> Optional[List[str]]:
    p = _cache_path(flores_code, split)
    if p.exists():
        return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return None


def _save_cache(flores_code: str, split: str, lines: Sequence[str]) -> None:
    p = _cache_path(flores_code, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


#: FLORES-200 is the preferred corpus but every current mirror is **gated**:
#: you must accept the terms on the dataset page and be logged in
#: (``huggingface-cli login``).  Parity therefore falls back to OPUS-100, which
#: is open, and records which one it used in ``Corpus.source`` so no result can
#: be attributed to the wrong corpus.
_FLORES_MIRRORS = ("openlanguagedata/flores_plus", "facebook/flores")
_GATED_HINT = (
    "FLORES-200 is gated on the Hub. To use it: accept the terms at "
    "https://huggingface.co/datasets/openlanguagedata/flores_plus and run `huggingface-cli login`. "
    "Falling back to OPUS-100 (open), which is a different corpus — the run manifest records which was used."
)
_gated_warned = False


def _download_flores(flores_code: str, split: str) -> Optional[List[str]]:
    """Try each known FLORES-200 mirror; return ``None`` if all fail.

    Claim: infrastructure.
    """
    global _gated_warned
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # pragma: no cover - datasets optional
        log.info("datasets not installed; cannot fetch FLORES")
        return None

    gated = False
    for repo in _FLORES_MIRRORS:
        try:
            ds = load_dataset(repo, flores_code, split=split)
        except Exception as exc:  # pragma: no cover - depends on network
            if "gated" in str(exc).lower():
                gated = True
            log.debug("FLORES fetch failed for %s/%s: %s", repo, flores_code, exc)
            continue
        for column in ("text", "sentence", f"sentence_{flores_code}"):
            if column in ds.column_names:
                lines = [str(x) for x in ds[column]]
                log.info("loaded FLORES %s/%s (%d lines) from %s", flores_code, split, len(lines), repo)
                return lines
        log.debug("FLORES repo %s had unexpected columns %s", repo, ds.column_names)
    if gated and not _gated_warned:
        _gated_warned = True
        log.warning(_GATED_HINT)
    return None


def load_local_parallel(lang: str) -> Optional[Tuple[List[str], List[str]]]:
    """Read a user-supplied parallel corpus from ``PARITY_PARALLEL_<LANG>``.

    Format: one pair per line, ``target<TAB>english``.  This is the escape hatch
    that matters: FLORES-200 is gated, OPUS-100 does not cover every language
    (Swahili, among others, is absent from its Hub mirror), and no fallback
    chain this repository ships will cover the long tail.  Rather than pretend
    otherwise, supplying your own corpus is one environment variable.

    Claim: reduction, infrastructure — keeps the meaning-controlled metric
    reachable for languages whose data nobody has packaged yet, which is
    precisely the set of languages this project is for.
    """
    path = os.environ.get(f"PARITY_PARALLEL_{lang.upper()}")
    if not path or not Path(path).exists():
        return None
    tgt, eng = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        t, e = line.split("\t", 1)
        if t.strip() and e.strip():
            tgt.append(t.strip())
            eng.append(e.strip())
    if not tgt:
        log.warning("PARITY_PARALLEL_%s=%s contained no 'target<TAB>english' lines", lang.upper(), path)
        return None
    log.info("loaded %d local parallel pairs for %s from %s", len(tgt), lang, path)
    return clean_bitext(tgt, eng)


def _download_opus100(
    lang: str, split: str = "validation", min_lines: int = 0
) -> Optional[Tuple[List[str], List[str]]]:
    """Fetch an open target↔English parallel split from OPUS-100.

    Returns ``(target_lines, english_lines)``, aligned pairwise.  OPUS-100 pairs
    each language with English separately, so the target sides of two languages
    are *not* row-aligned with each other — which is why
    :meth:`ParallelCorpus.pivot_for` exists.

    ``min_lines`` matters more than it looks.  OPUS-100's ``validation`` split is
    2 000 sentences: fine for *measuring* fertility, and far too small to
    *certify* a pack.  A (95%, 95%) tolerance limit needs 59 held-out occurrences
    of each token, and a certify slice cut from 2 000 lines does not contain them
    for anything but the commonest sequences.  So when the caller asks for more
    lines than a split holds, we go to ``train`` rather than silently returning a
    corpus on which every candidate will be refused for lack of evidence.

    Claim: reduction, bound — keeps the meaning-controlled measurement available
    without gated access, and makes the corpus large enough that a certificate
    can actually be computed.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # pragma: no cover
        return None
    # Not every pair has every split: the smaller pairs often ship train only.
    splits: List[str] = []
    if min_lines <= 2000:
        splits += [split, "test"]
    splits.append(f"train[:{max(2 * min_lines, 40000)}]")
    attempts = [(cfg, sp) for cfg in (f"en-{lang}", f"{lang}-en") for sp in splits]

    best: Optional[Tuple[List[str], List[str]]] = None
    for config, sp in attempts:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", config, split=sp)
        except Exception as exc:  # pragma: no cover - depends on network
            log.debug("opus-100 fetch failed for %s/%s: %s", config, sp, exc)
            continue
        if "translation" not in ds.column_names:
            continue
        tgt, eng = [], []
        for row in ds["translation"]:
            t, e = row.get(lang), row.get("en")
            if t and e and t.strip() and e.strip():
                tgt.append(str(t).strip())
                eng.append(str(e).strip())
        if not tgt:
            continue
        before = len(tgt)
        tgt, eng = clean_bitext(tgt, eng)
        log.info("loaded OPUS-100 %s/%s: %d pairs, %d after cleaning", config, sp, before, len(tgt))
        if best is None or len(tgt) > len(best[0]):
            best = (tgt, eng)
        if len(tgt) >= min_lines:
            return best
    return best


def clean_bitext(
    target: Sequence[str],
    english: Sequence[str],
    min_chars: int = 10,
    ratio_tolerance: float = 3.0,
) -> Tuple[List[str], List[str]]:
    """Drop misaligned or degenerate sentence pairs from a noisy bitext.

    Web-mined corpora such as OPUS-100 contain pairs that are not translations
    of each other, and a handful of them will dominate a corpus-level token
    ratio.  Standard bitext cleaning: require both sides to have real content,
    and drop pairs whose character-length ratio is more than ``ratio_tolerance``
    times away from the corpus median in either direction.

    Without this, the headline metric silently measures alignment noise instead
    of tokenizer behaviour — which is exactly the kind of number that looks
    authoritative and is not.  FLORES-200 needs none of this, which is why it is
    still preferred when access is available.

    Claim: reduction — protects the metric that every other number is relative
    to.
    """
    pairs = [(t, e) for t, e in zip(target, english) if len(t.strip()) >= min_chars and len(e.strip()) >= min_chars]
    if len(pairs) < 8:
        return [t for t, _ in pairs], [e for _, e in pairs]
    ratios = sorted(count_chars(t) / max(1, count_chars(e)) for t, e in pairs)
    median = ratios[len(ratios) // 2] or 1.0
    lo, hi = median / ratio_tolerance, median * ratio_tolerance
    kept = [(t, e) for t, e in pairs if lo <= count_chars(t) / max(1, count_chars(e)) <= hi]
    if len(kept) < len(pairs):
        log.debug("bitext cleaning removed %d/%d pairs", len(pairs) - len(kept), len(pairs))
    return [t for t, _ in kept], [e for _, e in kept]


def load_parallel(
    langs: Sequence[str],
    split: str = "dev",
    max_sentences: Optional[int] = None,
    allow_download: bool = True,
) -> ParallelCorpus:
    """Load an aligned parallel corpus for ``langs``, preferring FLORES-200.

    Falls back, per language and then globally, to the embedded sample.  A
    partial FLORES load is never mixed with the sample: alignment across
    languages must hold, so if any requested language is missing we fall back
    wholesale and say so in :attr:`ParallelCorpus.source`.

    Claim: reduction — supplies the meaning-controlled data on which every
    fertility and parity-ratio number in this repo is computed.
    """
    langs = list(dict.fromkeys(list(langs) + [PIVOT]))
    online = allow_download and os.environ.get("PARITY_OFFLINE") != "1"

    # -- 1. FLORES-200: fully aligned across every language ------------------
    flores: Dict[str, List[str]] = {}
    for code in langs:
        spec = language(code)
        lines = _load_cached(spec.flores, split)
        if lines is None and online:
            lines = _download_flores(spec.flores, split)
            if lines:
                _save_cache(spec.flores, split, lines)
        if not lines:
            flores = {}
            break
        flores[code] = lines
    if flores:
        n = min(len(v) for v in flores.values())
        if max_sentences:
            n = min(n, max_sentences)
        return ParallelCorpus(by_lang={k: v[:n] for k, v in flores.items()}, source="flores200", split=split)

    # -- 2. local override, then OPUS-100: pairwise aligned ------------------
    by_lang: Dict[str, List[str]] = {}
    pivots: Dict[str, List[str]] = {}
    unavailable: List[str] = []
    for code in langs:
        if code == PIVOT:
            continue
        pair = load_local_parallel(code)
        if pair:
            tgt, eng = pair
            if max_sentences:
                tgt, eng = tgt[:max_sentences], eng[:max_sentences]
            by_lang[code] = tgt
            pivots[code] = eng
    if online:
        for code in langs:
            if code == PIVOT or code in by_lang:
                continue
            cached_t = _load_cached(f"opus100.{code}", split)
            cached_e = _load_cached(f"opus100.{code}.en", split)
            pair = (
                (cached_t, cached_e)
                if cached_t and cached_e
                else _download_opus100(code, min_lines=max_sentences or 2000)
            )
            if not pair or not pair[0]:
                unavailable.append(code)
                continue
            tgt, eng = pair
            if not (cached_t and cached_e):
                _save_cache(f"opus100.{code}", split, tgt)
                _save_cache(f"opus100.{code}.en", split, eng)
            if max_sentences:
                tgt, eng = tgt[:max_sentences], eng[:max_sentences]
            by_lang[code] = tgt
            pivots[code] = eng

    if by_lang:
        # The English column is one language's pivot, kept so callers that ask
        # for `by_lang["en"]` still get real English text.
        first = next(iter(pivots))
        by_lang[PIVOT] = pivots[first]
        missing = [l for l in langs if l not in by_lang and l != PIVOT]
        if missing:
            log.warning(
                "no open parallel corpus for %s — omitted. FLORES-200 covers them but is gated; "
                "otherwise supply your own with PARITY_PARALLEL_%s=/path/to/pairs.tsv "
                "(one 'target<TAB>english' pair per line).",
                missing,
                missing[0].upper(),
            )
        source = "local+opus100" if any(os.environ.get(f"PARITY_PARALLEL_{c.upper()}") for c in by_lang) else "opus100"
        return ParallelCorpus(by_lang=by_lang, source=source, split=split, pivot_by_lang=pivots)

    # -- 3. the embedded sample ---------------------------------------------
    log.warning("no downloadable parallel corpus; falling back to the 24-sentence embedded sample")
    sample = load_embedded_sample([l for l in langs if l in load_embedded_sample().by_lang])
    if max_sentences:
        sample.by_lang = {k: v[:max_sentences] for k, v in sample.by_lang.items()}
    return sample


def load_monolingual(
    lang: str,
    split: str = "dev",
    max_sentences: Optional[int] = None,
    allow_download: bool = True,
) -> Corpus:
    """Load one language's lines — the mining corpus.

    Claim: reduction — the miner needs volume, not alignment, so this loader
    exists separately and can be pointed at any local text file via
    ``PARITY_MINING_CORPUS_<LANG>``.
    """
    override = os.environ.get(f"PARITY_MINING_CORPUS_{lang.upper()}")
    if override and Path(override).exists():
        lines = [ln for ln in Path(override).read_text(encoding="utf-8").splitlines() if ln.strip()]
        return Corpus(lang, lines[:max_sentences] if max_sentences else lines, source=f"file:{override}", split="custom")

    online = allow_download and os.environ.get("PARITY_OFFLINE") != "1"
    spec = language(lang)
    lines = _load_cached(spec.flores, split)
    if lines is None and online:
        lines = _download_flores(spec.flores, split)
        if lines:
            _save_cache(spec.flores, split, lines)
    if lines:
        return Corpus(lang, lines[:max_sentences] if max_sentences else lines, source="flores200", split=split)

    cached = _load_cached(f"opus100.{lang}", split)
    if cached is None and online:
        pair = _download_opus100(lang, min_lines=max_sentences or 2000)
        if pair:
            cached = pair[0]
            _save_cache(f"opus100.{lang}", split, pair[0])
            _save_cache(f"opus100.{lang}.en", split, pair[1])
    if cached:
        return Corpus(lang, cached[:max_sentences] if max_sentences else cached, source="opus100", split=split)

    sample = load_embedded_sample()
    if lang not in sample.by_lang:
        raise KeyError(f"no corpus available for {lang!r} offline; set PARITY_MINING_CORPUS_{lang.upper()}")
    lines = sample.by_lang[lang]
    return Corpus(lang, lines[:max_sentences] if max_sentences else lines, source="embedded_sample", split="sample")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def expand_for_testing(sentences: Sequence[str], n: int, seed: int = 0) -> List[str]:
    """Compose ``n`` distinct lines by pairing sentences — a **fixture only**.

    The offline sample has 24 sentences per language, which is too few to mine
    from and far too few to split four ways.  Pairing sentence ``i`` with
    sentence ``j`` yields up to ``24²`` distinct lines that share vocabulary the
    way real text does, so the mining/calibration machinery gets something to
    bite on and the splits stay line-disjoint.

    This is for exercising code paths.  No claim in the README is computed on
    it: :mod:`benchmarks.run` refuses to report headline numbers when
    ``Corpus.source`` is not a real corpus.

    Claim: infrastructure.
    """
    import random

    rng = random.Random(seed)
    m = len(sentences)
    pairs = [(i, j) for i in range(m) for j in range(m) if i != j]
    rng.shuffle(pairs)
    out = []
    for i, j in pairs[:n]:
        out.append(sentences[i] + " " + sentences[j])
    return out


def normalize(text: str) -> str:
    """NFC-normalise so that character counts are comparable across scripts.

    Without this, a Devanagari or Hangul string can be counted with a different
    number of code points depending on the producer's normalisation, which
    would move ``tokens_per_char`` for reasons that have nothing to do with the
    tokenizer.

    Claim: reduction — makes the denominator of the fertility metric honest.
    """
    return unicodedata.normalize("NFC", text)


def count_chars(text: str) -> int:
    """Count NFC code points, excluding whitespace.

    Whitespace is excluded because scripts differ in how much of it they use,
    and we are measuring the cost of *content*.

    Claim: reduction.
    """
    return sum(1 for ch in normalize(text) if not ch.isspace())


def count_words(text: str, lang: str) -> Optional[int]:
    """Count whitespace words, or ``None`` for unsegmented scripts.

    Returning ``None`` rather than a whitespace count for Japanese/Thai is a
    deliberate refusal: reporting "tokens per word" for a script without word
    spaces produces a number that looks comparable and is not.

    Claim: reduction.
    """
    spec = LANGUAGES.get(lang)
    if spec is not None and not spec.whitespace_delimited:
        return None
    words = [w for w in normalize(text).split() if w]
    return len(words)
