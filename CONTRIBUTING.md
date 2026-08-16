# Contributing

Two kinds of contribution, with different bars.

## Contributing a vocabulary pack for a language

This is the contribution the project exists for. It has its own guide:
[`docs/contributing-a-pack.md`](docs/contributing-a-pack.md).

The short version: point the CLI at a corpus you trust, **read the mined tokens
as a speaker of the language**, and open a pull request with the pack and the
build command. Review by speakers is part of the process, not a formality — a
merge table that splits words in places that make no sense to the people who use
them is the problem this project started from.

You do not need to understand the least-squares solver to do this.

## Contributing code

```bash
git clone https://github.com/NagaYu/parity
cd parity
pip install -e ".[dev,models,bench]"
pytest -q                    # ~40s, no network, no model downloads
```

### Two rules the test suite enforces

**1. Every public callable declares which claim it substantiates.** Each
docstring ends with a `Claim:` line naming one or more of `reduction`,
`non-regression`, `bound`, `low-cost`, `infrastructure`.
`tests/test_docstring_claims.py` fails the build if one is missing. This is not
decoration: it is how a reader knows whether a function is load-bearing for a
guarantee or is plumbing.

**2. The suite runs fully offline.** CI sets `PARITY_OFFLINE=1` and
`HF_HUB_OFFLINE=1`. Anything that needs a download must degrade gracefully to
`parity/tiny.py` (a real 200k-parameter decoder) and the 24-sentence embedded
corpus. A structural guarantee that can only be checked by downloading a billion
parameters will not be checked.

### What a good change looks like here

- **If it touches a bound**, it comes with a test that could falsify it. The
  certificate machinery is the point of the project; a change that makes bounds
  *look* better without evidence is worse than no change.
- **If it adds a metric**, it says what denominator it uses and refuses to
  compute where that denominator is meaningless. See
  `parity.corpora.count_words`, which returns `None` for scripts without word
  spaces rather than silently counting whitespace.
- **If it reports a number**, provenance travels with it: `measured` vs
  `extrapolated`, and which corpus. `benchmarks/report.py` and
  `figures/make_pareto.py` both mark extrapolated figures, and the figure
  watermarks fixture runs *inside the image*.

### Language and framing

[`docs/framing.md`](docs/framing.md) is a constraint on the code and every string
it emits, not a disclaimer. Token cost is a property of a tokenizer; it is not a
property of a language, and nothing in this repository — docstrings, log
messages, model cards, commit messages — should read as though it were. That
document lists the specific phrasings to avoid and the three places where the
framing changes what the code actually does.

### Things that would help

- Packs for more languages, especially ones OPUS-100 does not cover.
- Script-specific normalisation. `parity.corpora.normalize` is NFC for
  everything, which is not right for every script. We would rather be told than
  guess.
- A vLLM integration test. `serving/vllm_plugin.py` is tested against its own
  contract but never against a running vLLM engine.
- Tighter certificates. The conformal tail bound is honest but loose; a
  variance-adaptive or PAC-Bayes treatment could adopt more tokens at the same
  risk.

### Things that are out of scope

- Anything that puts the adaptation burden on the user rather than the system
  (advice to write shorter prompts, switch languages, and so on).
- Measurement without capability. It is easy to add another dashboard showing
  that some languages cost more; that changes nothing, and the deliverable here
  is the reduction and the bound on what it costs in behaviour.
