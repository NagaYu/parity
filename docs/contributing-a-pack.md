# Contributing a vocabulary pack for your language

You do not need to understand the least-squares solver to contribute a pack. You
need a corpus you trust, and the willingness to read a list of tokens and say
whether they look like units of your language.

That second part is the part that cannot be automated, and it is why this
document exists.

## 0. What you are producing

A directory:

```
packs/<lang>/
  manifest.json           tokens, ids, certificates, provenance
  embeddings.safetensors  the new rows
  added_tokens.json       surface strings, for interoperability
  README.md               the model card — generated, never hand-written
```

It attaches to one specific base model. Its tokenizer fingerprint is checked at
load time, so a pack cannot be attached to the wrong model by accident.

## 1. Get a corpus you trust

FLORES-200 is the default and is enough to *measure* the problem, but not to
build a large pack — see step 5. For a real pack you want on the order of 10⁷
tokens of text you would be comfortable having a model shaped by.

```bash
export PARITY_MINING_CORPUS_TE=/path/to/telugu.txt   # one document per line
```

For the *measurement* side you also want a parallel corpus. FLORES-200 is the best
option but is gated on the Hub — accept the terms on the dataset page and run
`huggingface-cli login`. Failing that, Parity falls back to OPUS-100 (open, but
web-mined and not covering every language). If neither has your language, supply
your own pairs:

```bash
export PARITY_PARALLEL_TE=/path/to/pairs.tsv        # target<TAB>english per line
```

A few hundred good pairs are worth more here than tens of thousands of noisy ones:
`parity.corpora.clean_bitext` will drop badly-aligned pairs, but it cannot repair
translations that are merely wrong.

Things worth checking before you use a corpus, because the mined tokens will
reflect all of them:

- **Domain.** A pack mined on news will save tokens on news. If your users write
  conversationally, mine conversation.
- **Register and orthography.** If your language has multiple accepted
  orthographies or a formal/colloquial split, decide which one this pack is for,
  and say so in the pull request. It is fine to ship two packs.
- **Machine translation.** Corpora scraped from the web are frequently MT output.
  Tokens mined from bad MT will encode bad MT.
- **Provenance and consent.** If the corpus is not something you have the right
  to use, the pack inherits that problem.

## 2. Look at the baseline

```bash
parity atlas --model Qwen/Qwen2.5-0.5B-Instruct --langs te,en
```

This tells you what the tokenizer currently charges your language relative to
English, and therefore how much headroom a pack has.

## 3. Build

```bash
parity build \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --lang te \
  --budget 8000 \
  --max-kl 0.05 \
  --out packs/te
```

What the flags mean in practice:

| flag | meaning |
| --- | --- |
| `--budget` | how many embedding rows you are willing to add (`budget × d_model` parameters) |
| `--max-kl` | how far you will let the model's next-token distribution move, in nats |
| `--alpha` / `--delta` | the tolerance limit: "≥ 1−α of inputs, with confidence 1−δ" |
| `--max-offcontext-mass` | how much probability a new token may take where it does not belong |

Lower tolerances mean fewer tokens adopted, not weaker guarantees. Parity never
ships a token it could not certify.

## 4. **Read the tokens.** This step is not optional

```bash
parity inspect --pack packs/te --limit 200
```

You will see the surface string of every adopted token. Read them as a speaker,
not as an engineer, and ask:

- Are these **units**? Morphemes, common words, frequent affixes, natural
  collocations — or arbitrary fragments that happen to be frequent?
- Do any of them cut **across** a morpheme boundary in a way that would look
  wrong to a reader?
- Is anything in there a **slur, a stereotype, or a politically loaded phrase**
  that the corpus happened to repeat? Frequency is not endorsement, and a token
  is a permanent, first-class part of the model's vocabulary.
- Does the list over-represent one **dialect, region, or register** at the
  expense of others your users write in?

If the list looks wrong, the corpus is usually the reason. Change the corpus
before changing the tolerances.

Please get at least one other speaker to look too. A pull request that says "two
of us read the token list" carries more weight here than one that says "the
certificate passed".

## 5. Check the refusals

`parity build` prints why candidates were refused, and the two reasons call for
opposite fixes:

- **"insufficient held-out occurrences"** — not a drift problem. Your calibration
  corpus is too small for the requested (α, δ). Get more text, or accept a weaker
  guarantee explicitly with `--alpha`/`--delta`. A (95%, 95%) limit needs 59
  held-out occurrences of *each* token.
- **"KL / TV / emission / off-context drift above tolerance"** — genuine drift.
  These candidates are ones the model cannot represent as a single token without
  moving. Leaving them out is the system working.

## 6. Verify, independently

```bash
parity verify --pack packs/te --model Qwen/Qwen2.5-0.5B-Instruct
```

This re-measures drift on fresh contexts and checks the realised coverage against
what each bound promised. Run it on data the build never saw. If it fails, say so
in the pull request — a failing verification is information, not an embarrassment.

## 7. Publish

```bash
python scripts/push_vocab_pack.py --pack packs/te --repo your-org/parity-te-qwen2.5-0.5b --dry-run
```

`--dry-run` writes the card without uploading anything, so you can check the
numbers first. The card is generated from the manifest — the certified drift on
the card is the certified drift in the evidence, and there is no code path that
publishes a hand-written one.

## 8. Open the pull request

Include:

- the corpus: what it is, where it came from, why you trust it;
- the build command, verbatim;
- `parity inspect` output, or a representative slice of it;
- who reviewed the token list, and what they changed;
- the certified bounds, and anything the verification step flagged;
- anything you deliberately excluded, and why.

## Questions that are welcome

- "My language has no widely agreed word segmentation." Fine — Parity works in
  token-id space and never needs word boundaries. `tokens_per_word` will report
  `None` for your script, which is deliberate.
- "My script needs normalisation that this repo does not do." Open an issue.
  `parity.corpora.normalize` is currently NFC for everything, which is not right
  for every script, and we would rather be told than guess.
- "The mined tokens look fine but the reduction is small." That is a real result
  and worth reporting. It usually means the base tokenizer already covers your
  script reasonably, which is good news for your users.
- "I want a pack for a language not in `parity.corpora.LANGUAGES`." Add it — the
  registry is one dataclass entry, and the only field that needs thought is
  `whitespace_delimited`.
