"""MultiTokenizerServing — one model instance, one weight set, many tokenizers.

The deployment story Parity is built for: a provider hosts `Qwen2.5-1.5B` once
and serves English requests with the original tokenizer while serving Japanese,
Hindi, Arabic, Thai and Swahili requests with their vocabulary packs — in the
same process, in the same batch, against the same weights.

Why that is cheap
-----------------
Three structural facts, none of which is a heuristic:

1. **One id space.**  Pack ids are appended after the base vocabulary, so an id
   identifies the same token in every view.  The model does not know or care
   which view produced its input.

2. **One weight set.**  A pack only adds embedding rows.  Batching a base-view
   request next to a pack-view request is not "two models in a batch"; it is one
   model, and the batch is exactly as efficient as a single-view batch.

3. **View isolation happens at the edges.**  A view constrains what the
   *tokenizer* may emit and what the *sampler* may select.  Both are O(vocab)
   per step at most, and the logit mask is precomputed once per view.

So the incremental cost of serving N views is a per-request tokenizer dispatch
and a mask select.  :meth:`MultiTokenizerRouter.benchmark` measures exactly that
and reports it separately from model time, which is benchmark metric (6).

Why English stays exactly English
---------------------------------
For a base-view request: the input contains only base ids, so the hidden states
are computed from unmodified embedding rows; the logits at base positions come
from unmodified unembedding rows; and the mask sets every pack logit to −∞
before the softmax.  Masked-softmax over the base subset equals
``exp(z_i) / Σ_{j ∈ base} exp(z_j)`` — the original model's distribution, to the
last bit.  This is an identity, not an empirical finding, and
``tests/test_english_nonregression.py`` asserts it numerically anyway.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from parity.adapters import TorchLMAdapter
from parity.serving.prefix_cache import PrefixCache, clone_kv, crop_kv, kv_length
from parity.tokenization import BASE_VIEW, AugmentedTokenizer, TokenizerView

log = logging.getLogger("parity.serving")

NEG_INF = float("-inf")


@dataclass
class Request:
    """One inference request, carrying the view it wants.

    Claim: infrastructure — the view is a request-level property, which is the
    whole point: a provider does not need one deployment per language.
    """

    prompt: str
    view: str = "base"
    max_new_tokens: int = 16
    request_id: str = ""


@dataclass
class Response:
    """Result of one request, with the accounting a provider would bill from.

    Claim: reduction, low-cost.
    """

    text: str
    prompt_ids: List[int]
    output_ids: List[int]
    view: str
    prompt_tokens: int
    generated_tokens: int
    cached_prefix_tokens: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Tokens billed for this request.

        Claim: reduction.
        """
        return self.prompt_tokens + self.generated_tokens


@dataclass
class ThroughputReport:
    """Timing breakdown for a mixed-view workload.

    Separating tokenizer/mask time from model time is what turns "the overhead
    is small" into a number that can be checked: the model term is identical
    between the single-view and multi-view runs by construction, so any
    difference lives entirely in the other two.

    Claim: low-cost — benchmark metric (6).
    """

    label: str
    n_requests: int
    total_tokens: int
    wall_seconds: float
    tokenizer_seconds: float
    mask_seconds: float
    model_seconds: float
    cache: Dict[str, float] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """End-to-end throughput.

        Claim: low-cost.
        """
        return self.total_tokens / max(1e-9, self.wall_seconds)

    @property
    def dispatch_overhead(self) -> float:
        """Share of wall-clock spent on view dispatch rather than the model.

        Claim: low-cost — the quantity metric (6) asks to be small.
        """
        return (self.tokenizer_seconds + self.mask_seconds) / max(1e-9, self.wall_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise.

        Claim: infrastructure.
        """
        return {
            "label": self.label,
            "n_requests": self.n_requests,
            "total_tokens": self.total_tokens,
            "wall_seconds": self.wall_seconds,
            "tokens_per_second": self.tokens_per_second,
            "tokenizer_seconds": self.tokenizer_seconds,
            "mask_seconds": self.mask_seconds,
            "model_seconds": self.model_seconds,
            "dispatch_overhead": self.dispatch_overhead,
            "cache": self.cache,
        }


class MultiTokenizerRouter:
    """Serve many tokenizer views from a single model instance.

    Claim: non-regression, low-cost, reduction — the deployment surface that
    makes the other three claims usable by anyone other than the author.
    """

    def __init__(
        self,
        adapter: TorchLMAdapter,
        tokenizer: AugmentedTokenizer,
        cache: Optional[PrefixCache] = None,
        pad_id: int = 0,
    ):
        self.lm = adapter
        self.tok = tokenizer
        self.cache = cache if cache is not None else PrefixCache()
        self.pad_id = pad_id
        self._masks: Dict[str, torch.Tensor] = {}
        self._views: Dict[str, TokenizerView] = {"base": BASE_VIEW}
        for lang in tokenizer.packs():
            self._views[lang] = tokenizer.view(lang)
        if adapter.vocab_size() != tokenizer.total_vocab_size:
            raise ValueError(
                f"model has {adapter.vocab_size()} embedding rows but the tokenizer expects "
                f"{tokenizer.total_vocab_size}; attach packs to both or neither"
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        packs: Sequence[str] = (),
        device: str = "cpu",
        dtype: str = "float32",
    ) -> "MultiTokenizerRouter":
        """Load a base model plus zero or more packs from disk or the Hub.

        Claim: infrastructure — the three-line adoption path advertised in the
        README resolves to this.
        """
        from parity.pack import load_pack
        from parity.tokenization import HFTokenizer

        adapter = TorchLMAdapter.from_pretrained(model_id, dtype=dtype, device=device)
        tokenizer = HFTokenizer.from_pretrained(model_id)
        aug = AugmentedTokenizer(tokenizer)
        for p in packs:
            pack = load_pack(p)
            aug.attach(pack)
            adapter.append_rows(pack.input_embeddings, pack.output_embeddings)
        aug.check_invariants()
        return cls(adapter, aug)

    # -- views --------------------------------------------------------------

    def views(self) -> List[str]:
        """Names of every available view, ``"base"`` first.

        Claim: infrastructure.
        """
        return ["base"] + sorted(v for v in self._views if v != "base")

    def view(self, name: str) -> TokenizerView:
        """Look up a view by name.

        Claim: infrastructure.
        """
        if name not in self._views:
            raise KeyError(f"unknown view {name!r}; available: {self.views()}")
        return self._views[name]

    def logit_mask(self, name: str) -> torch.Tensor:
        """Additive logit mask for a view: ``0`` on allowed ids, ``−inf`` elsewhere.

        Computed once per view and cached.  Applying it before the softmax is
        what makes a base-view request produce exactly the original model's
        distribution, and what stops a Japanese-pack token leaking into an
        English response.

        Claim: non-regression.
        """
        if name in self._masks:
            return self._masks[name]
        V = self.tok.total_vocab_size
        mask = torch.zeros(V, dtype=torch.float32)
        v = self.view(name)
        if v.allowed_ids is None:
            mask[self.tok.base_vocab_size :] = NEG_INF
        else:
            allowed = torch.zeros(V, dtype=torch.bool)
            allowed[torch.tensor(sorted(v.allowed_ids), dtype=torch.long)] = True
            mask[~allowed] = NEG_INF
        self._masks[name] = mask
        return mask

    # -- tokenization -------------------------------------------------------

    def encode(self, text: str, view: str = "base") -> List[int]:
        """Encode under a named view.

        Claim: reduction.
        """
        return self.tok.encode(text, self.view(view))

    def decode(self, ids: Sequence[int]) -> str:
        """Decode ids, expanding pack tokens; view-independent by design.

        Claim: non-regression.
        """
        return self.tok.decode(ids)

    def count(self, text: str, view: str = "base") -> int:
        """Token count under a view — what the request would be billed.

        Claim: reduction.
        """
        return len(self.encode(text, view))

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def logits(self, ids: Sequence[int], view: str = "base") -> torch.Tensor:
        """Masked next-token logits for a full id sequence.

        Claim: non-regression.
        """
        t = torch.as_tensor(list(ids), dtype=torch.long, device=self.lm.device)[None, :]
        out, _ = self.lm.run(t, use_cache=False)
        return out[0, -1].float() + self.logit_mask(view)

    @torch.no_grad()
    def generate(self, request: Request, use_cache: bool = True) -> Response:
        """Greedy generation for a single request, using the shared prefix cache.

        Greedy rather than sampled so that the multi-view correctness test is
        exact rather than distributional.

        Claim: non-regression, low-cost.
        """
        t0 = time.time()
        ids = self.encode(request.prompt, request.view)
        prompt_len = len(ids)
        mask = self.logit_mask(request.view)

        past = None
        start = 0
        reused = 0
        if use_cache and prompt_len > 1:
            entry, matched = self.cache.lookup(ids[:-1], request.view)
            if entry is not None and matched > 0:
                past = crop_kv(clone_kv(entry.kv), matched)
                start = matched
                reused = matched

        cur = torch.as_tensor(ids[start:], dtype=torch.long, device=self.lm.device)[None, :]
        pos = torch.arange(start, prompt_len, device=self.lm.device)[None, :]
        attn = torch.ones((1, prompt_len), dtype=torch.long, device=self.lm.device)
        out, past = self.lm.run(cur, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
        if use_cache and prompt_len - 1 >= self.cache.min_prefix:
            self.cache.insert(ids[:-1], crop_kv(clone_kv(past), prompt_len - 1), request.view)

        generated: List[int] = []
        for step in range(request.max_new_tokens):
            nxt = int(torch.argmax(out[0, -1].float() + mask))
            generated.append(nxt)
            if step + 1 >= request.max_new_tokens:
                break
            nid = torch.tensor([[nxt]], dtype=torch.long, device=self.lm.device)
            p = prompt_len + step
            attn = torch.ones((1, p + 1), dtype=torch.long, device=self.lm.device)
            out, past = self.lm.run(
                nid,
                attention_mask=attn,
                position_ids=torch.tensor([[p]], device=self.lm.device),
                past_key_values=past,
                use_cache=True,
            )

        return Response(
            text=self.decode(generated),
            prompt_ids=ids,
            output_ids=generated,
            view=request.view,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated),
            cached_prefix_tokens=reused,
            seconds=time.time() - t0,
        )

    @torch.no_grad()
    def batch_generate(self, requests: Sequence[Request]) -> List[Response]:
        """Greedy generation for a batch whose requests may use different views.

        All requests share one forward pass per step.  Left padding plus
        explicit ``position_ids`` aligns their last real token at the same slot;
        each request's own logit mask is applied to its own row before the
        argmax.  There is no per-view weight switching because there are no
        per-view weights.

        Claim: non-regression, low-cost — the concrete demonstration that mixed
        views batch together, which is what metric (6) is about.
        """
        if not requests:
            return []
        t0 = time.time()
        enc = [self.encode(r.prompt, r.view) for r in requests]
        B = len(requests)
        T = max(len(e) for e in enc)
        dev = self.lm.device

        ids = torch.full((B, T), self.pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((B, T), dtype=torch.long, device=dev)
        for i, e in enumerate(enc):
            ids[i, T - len(e) :] = torch.as_tensor(e, dtype=torch.long, device=dev)
            attn[i, T - len(e) :] = 1
        pos = (attn.cumsum(dim=1) - 1).clamp_min(0)

        masks = torch.stack([self.logit_mask(r.view) for r in requests])  # [B, V]
        out, past = self.lm.run(ids, attention_mask=attn, position_ids=pos, past_key_values=None, use_cache=True)

        max_new = max(r.max_new_tokens for r in requests)
        generated: List[List[int]] = [[] for _ in range(B)]
        last_pos = pos[:, -1]
        for step in range(max_new):
            nxt = torch.argmax(out[:, -1].float() + masks, dim=-1)  # [B]
            for i, r in enumerate(requests):
                if step < r.max_new_tokens:
                    generated[i].append(int(nxt[i]))
            if step + 1 >= max_new:
                break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
            last_pos = last_pos + 1
            out, past = self.lm.run(
                nxt[:, None],
                attention_mask=attn,
                position_ids=last_pos[:, None],
                past_key_values=past,
                use_cache=True,
            )

        secs = time.time() - t0
        return [
            Response(
                text=self.decode(generated[i]),
                prompt_ids=enc[i],
                output_ids=generated[i],
                view=r.view,
                prompt_tokens=len(enc[i]),
                generated_tokens=len(generated[i]),
                seconds=secs / B,
            )
            for i, r in enumerate(requests)
        ]

    # -- measurement --------------------------------------------------------

    @torch.no_grad()
    def benchmark(self, requests: Sequence[Request], label: str = "mixed", batched: bool = True) -> ThroughputReport:
        """Time a workload and split the cost into tokenizer / mask / model.

        Run it twice — once with the requests' real views and once with every
        request forced to ``base`` — to obtain the multi-tokenizer throughput
        degradation directly, with the model term held fixed.

        Claim: low-cost — benchmark metric (6).
        """
        self.cache.clear()
        tok_s = mask_s = 0.0

        t_tok = time.time()
        enc = [self.encode(r.prompt, r.view) for r in requests]
        tok_s += time.time() - t_tok

        t_mask = time.time()
        _ = torch.stack([self.logit_mask(r.view) for r in requests])
        mask_s += time.time() - t_mask

        t0 = time.time()
        if batched:
            responses = self.batch_generate(requests)
        else:
            responses = [self.generate(r) for r in requests]
        wall = time.time() - t0 + tok_s + mask_s
        total = sum(r.total_tokens for r in responses)
        return ThroughputReport(
            label=label,
            n_requests=len(requests),
            total_tokens=total,
            wall_seconds=wall,
            tokenizer_seconds=tok_s,
            mask_seconds=mask_s,
            model_seconds=max(0.0, wall - tok_s - mask_s),
            cache=self.cache.stats.to_dict(),
        )

    def compare_single_vs_multi(self, requests: Sequence[Request], batched: bool = True) -> Dict[str, Any]:
        """Throughput with real views vs. the same workload forced to base.

        Claim: low-cost — the direct answer to "how much does serving several
        tokenizers at once cost you?", reported as a ratio rather than a vibe.
        """
        multi = self.benchmark(requests, label="multi-view", batched=batched)
        single = self.benchmark(
            [Request(r.prompt, "base", r.max_new_tokens, r.request_id) for r in requests],
            label="base-only",
            batched=batched,
        )
        return {
            "multi_view": multi.to_dict(),
            "base_only": single.to_dict(),
            # <1 means multi-view is slower in tokens/s; note that the multi-view
            # run also processes *fewer* tokens for the same text, which is the
            # point — see 'effective_speedup'.
            "throughput_ratio": multi.tokens_per_second / max(1e-9, single.tokens_per_second),
            "token_ratio": multi.total_tokens / max(1, single.total_tokens),
            "effective_speedup": (single.wall_seconds / max(1e-9, multi.wall_seconds)),
            "dispatch_overhead": multi.dispatch_overhead,
        }
