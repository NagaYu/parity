# Framing: what this project says, and what it refuses to say

This is a constraint on the code and on every string it emits, not a disclaimer
appended at the end. It is short enough to read before changing anything here.

## The rule

**Token cost is a property of a tokenizer. It is not a property of a language.**

A tokenizer is fitted to a corpus. The corpora used to fit today's tokenizers
under-represented most of the world's writing systems. The resulting merge table
covers English strings densely and, say, Telugu strings barely at all, so Telugu
text falls back toward byte level and costs three to five times more tokens for
the same meaning. That is arithmetic about an artefact and its training data.

Every number in this repository is a measurement of that artefact.

## Language that is not used here

Not in docstrings, not in log messages, not in the Space, not in a model card,
not in a commit message:

| avoid | why | instead |
| --- | --- | --- |
| "inefficient language" | attributes an engineering defect to a people | "high token cost under this tokenizer" |
| "verbose script" | scripts encode information differently; none is padded | "more code points per unit of meaning" |
| "hard language for LLMs" | the difficulty is in the tooling | "under-served by this tokenizer" |
| "fixing Japanese" | Japanese is not broken | "fixing the tokenizer's coverage of Japanese" |
| "exotic", "unusual", "non-standard" script | standard to ~1.5 billion people | name the script |
| "low-resource language" (as an inherent property) | resources are an artefact of collection choices | "under-collected", "under-represented in the training corpus" |

The pattern to watch for: any phrasing where the *language* is the subject of a
negative predicate. The tokenizer should be the subject.

Two acceptable, precise uses:

- **"fertility"** — the established technical term for tokens per word. Used
  with its denominator stated, and refused entirely for scripts without word
  spaces (see `parity.corpora.count_words`, which returns `None` rather than a
  misleading whitespace count for Japanese and Thai).
- **"under-served"** — describes the tooling's relationship to the language,
  with the tooling as the agent.

## Why this is a technical constraint, not a style preference

Three places where the framing changes what the code does:

1. **Metric choice.** `tokens_per_word` is *refused*, not approximated, for
   unsegmented scripts. Reporting a whitespace word count for Japanese would
   produce a number that looks comparable to English and is not. The primary
   metric is `parity_ratio` — tokens per English-equivalent sentence on a
   parallel corpus — precisely because it is well-defined for every script.

2. **Where the burden falls.** Parity does not ask speakers to write differently,
   shorten their prompts, or switch to English. It changes the tokenizer. If a
   proposed feature would put the adaptation burden on the user rather than the
   system, it is the wrong feature.

3. **Who reviews a pack.** Mined tokens are printed as *strings*, not ids, and
   review by speakers of the language is part of the contribution process
   (`docs/contributing-a-pack.md`). A merge table that splits words in places
   that make no sense to the people who use them is the problem this project
   started from. Reproducing it faster, in a new file format, would not be
   progress.

## Measurement is the smallest part

It is easy to build a dashboard showing that some languages cost 5x more, and
such a dashboard changes nothing. This repository ships the measurement (the
atlas) because a baseline has to be auditable, but the deliverable is
`parity build` — the reduction and the bound on what the reduction costs in
behaviour.

If a change to this repository adds measurement without adding capability, it is
probably going in the wrong direction.

## Scope, stated where it can be seen

Parity applies to open-weight models and to the providers serving them; it needs
access to the embedding matrix. It cannot be applied from outside a closed API.

This is stated in the README, in the Space, and here, rather than left implicit,
because the people most affected by token-cost asymmetry are often the least able
to act on it — they are API consumers, not model hosts. Pretending otherwise
would be a second way of putting the burden in the wrong place.
