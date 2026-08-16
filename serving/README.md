# Serving Parity packs

One model instance, one weight set, many tokenizer views.

The core router lives in [`parity/serving/`](../parity/serving) and is engine-agnostic.
The two adapters here wire it into vLLM and TGI. They share one implementation of
view isolation rather than reimplementing it three times, because the safety
property ("an English request cannot see a Japanese pack token") is exactly the
kind of thing that rots when it is written three times.

## Why this is cheap

| what changes per view | cost |
| --- | --- |
| tokenizer used to encode the prompt | one dict lookup + one merge pass |
| logit mask applied before sampling | one precomputed `[vocab]` add |
| model weights | **nothing** |
| KV cache layout | **nothing** |

Pack token ids are appended after the base vocabulary, so an id means the same
thing in every view. Two consequences that matter operationally:

1. **Requests on different views batch together.** They are not "two models in a
   batch"; they are one model. Throughput is unchanged.
2. **The prefix cache is not fragmented by view.** A KV entry is a function of
   the token-id prefix and the weights, both of which are shared. An entry
   produced under one view is numerically valid for any request whose id prefix
   matches — including a base-view request. `parity/serving/prefix_cache.py`
   counts these as `cross_view_hits`, and `tests/test_serving.py` asserts the
   reused result is identical to a cold run.

The only thing a view restricts is what may be *emitted*, which is a sampler
concern and never touches the cache.

## vLLM

```bash
pip install vllm
python -m serving.vllm_plugin --model Qwen/Qwen2.5-0.5B-Instruct --packs packs/ja packs/hi
```

`vllm_plugin.py` does three things:

1. `patch_model(llm, packs)` — resizes the embedding matrix and writes the pack
   rows in place, asserting that no pre-existing row changed.
2. `ParityTokenizerRegistry` — resolves a request's `view` to the right
   `AugmentedTokenizer` before the scheduler sees the prompt.
3. `ViewLogitsProcessor` — a per-request `LogitsProcessor` carrying the view's
   additive mask.

vLLM's public API moves between releases, so the plugin imports it lazily and
each integration point is a small function you can re-point at the current API.
`python -m serving.vllm_plugin --check` reports what it found without loading a
model.

## TGI

TGI does tokenization in its Rust router, so the integration point is different:
run `serving/tgi_plugin.py` as a sidecar that tokenizes with the requested view
and submits pre-tokenized `input_ids` to TGI, and apply the mask through the
`grammar`/logit-processor hook. See the module docstring for the exact
request shape.

## Doing it yourself

Any engine that lets you (a) supply token ids instead of text and (b) add a
per-request logits processor can serve Parity packs. That is the whole
requirement:

```python
from parity import serving

router = serving.load("Qwen/Qwen2.5-0.5B-Instruct", packs=["packs/ja"])
ids  = router.encode(prompt, view="ja")     # fewer tokens
mask = router.logit_mask("ja")              # add to logits before sampling
```
