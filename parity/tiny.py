"""A small self-contained causal LM, so the whole pipeline is testable offline.

Parity's guarantees are structural — append-only embedding rows, view isolation,
a least-squares fit on a frozen network — and structural claims should be
testable without downloading 1B parameters.  ``TinyCausalLM`` is a real decoder
(RMSNorm, RoPE, causal MHA, SwiGLU, optional weight tying) at ~200k parameters,
which exercises every code path that a Qwen or Llama checkpoint does.

It is **not** a language model anyone should use for language.  Its role is to
make ``pytest`` a genuine gate: if view isolation breaks, or an embedding row is
written where it should not be, these tests fail in two seconds with no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    """Shape of a :class:`TinyCausalLM`.

    Claim: infrastructure.
    """

    vocab_size: int = 512
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 128
    max_seq: int = 512
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used by Llama/Qwen.

    Claim: infrastructure — matching the real architecture matters because the
    synthesis objective is sensitive to whether the first operation applied to
    an embedding is affine (LayerNorm) or scale-only (RMSNorm).
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise by RMS and rescale.

        Claim: infrastructure.
        """
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def _rope_from_positions(position_ids: torch.Tensor, head_dim: int, theta: float, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin for explicit per-sequence positions, shaped ``[B, 1, T, Dh/2]``.

    Per-sequence positions (rather than ``arange``) are what make **left**
    padding correct, and left padding is what makes a mixed-view batch possible
    — every request in the batch ends at the same slot, so one decode step
    advances them all.  Serving several tokenizer views from one model is only
    cheap if they can share a batch, so this detail carries the throughput claim.

    Claim: low-cost, infrastructure.
    """
    device = position_ids.device
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float()[..., None] * inv[None, None, :]  # [B, T, Dh/2]
    return freqs.cos().to(dtype)[:, None], freqs.sin().to(dtype)[:, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]; cos/sin: [B, 1, T, Dh/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Block(nn.Module):
    """One pre-norm transformer block with RoPE attention and SwiGLU MLP.

    Claim: infrastructure.
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.attn_norm = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        attn_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Residual update for one block, with optional KV reuse.

        The cache path exists so that :mod:`parity.serving.prefix_cache` can be
        tested for real: the claim that a prefix cache stays valid across
        tokenizer views is only worth something if there is an actual cache to
        reuse.

        Claim: infrastructure, low-cost.
        """
        B, T, D = x.shape
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None
        Tk = k.shape[2]
        past_len = Tk - T
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        qi = torch.arange(T, device=x.device)[:, None] + past_len
        kj = torch.arange(Tk, device=x.device)[None, :]
        causal = kj <= qi
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attn_mask is not None:
            pad = attn_mask[:, None, None, :Tk].to(torch.bool)
            scores = scores.masked_fill(~pad, torch.finfo(scores.dtype).min)
        att = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        h = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return (x, new_kv) if use_cache else (x, None)


@dataclass
class TinyOutput:
    """``transformers``-shaped forward output.

    Claim: infrastructure — the adapter can then be written once for both.
    """

    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class TinyCausalLM(nn.Module):
    """Decoder-only LM with the same call surface as an HF causal LM.

    Accepts ``input_ids`` **or** ``inputs_embeds``; the latter is what lets
    :mod:`parity.synthesis` optimise a candidate embedding without ever writing
    it into the weight matrix.

    Claim: infrastructure, low-cost — makes the full build/certify/serve loop
    runnable in a unit test.
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """HF-compatible accessor.

        Claim: infrastructure.
        """
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """HF-compatible accessor.

        Claim: infrastructure.
        """
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        **_,
    ) -> TinyOutput:
        """Run the model, optionally returning every layer's residual stream.

        Claim: infrastructure.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds
        B, T, D = x.shape
        past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=x.device)[None, :].expand(B, T)
        cos, sin = _rope_from_positions(
            position_ids, self.config.d_model // self.config.n_heads, self.config.rope_theta, x.dtype
        )
        # Hidden-state indexing mirrors transformers exactly: index 0 is the
        # embedding output, index i is the input to block i, and the final entry
        # is *after* the last norm — i.e. the vector the unembedding reads.
        # parity.certificate's Lipschitz bound depends on that last property.
        hs: List[torch.Tensor] = []
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for li, blk in enumerate(self.blocks):
            if output_hidden_states:
                hs.append(x)
            past = past_key_values[li] if past_key_values else None
            x, kv = blk(x, cos, sin, attention_mask, past_kv=past, use_cache=use_cache)
            if use_cache and kv is not None:
                new_cache.append(kv)
        x = self.norm(x)
        if output_hidden_states:
            hs.append(x)
        logits = self.lm_head(x)
        return TinyOutput(
            logits=logits,
            hidden_states=tuple(hs) if output_hidden_states else None,
            past_key_values=new_cache if use_cache else None,
        )


def build_tiny_model(vocab_size: int, seed: int = 0, tie: bool = False, **kw) -> TinyCausalLM:
    """Deterministically construct a small model for tests and demos.

    The embedding matrix is given a mild low-rank + noise structure rather than
    pure i.i.d. noise, because pure noise makes sub-token composition trivially
    optimal and would flatter the synthesis step.

    Claim: infrastructure — a fixture that does not accidentally prove the
    result it is supposed to test.
    """
    torch.manual_seed(seed)
    cfg = TinyConfig(vocab_size=vocab_size, tie_word_embeddings=tie, **kw)
    model = TinyCausalLM(cfg)
    with torch.no_grad():
        d = cfg.d_model
        rank = max(4, d // 8)
        basis = torch.randn(rank, d) / math.sqrt(d)
        coeff = torch.randn(vocab_size, rank)
        emb = coeff @ basis + 0.35 * torch.randn(vocab_size, d) / math.sqrt(d)
        model.embed_tokens.weight.copy_(emb)
        if not cfg.tie_word_embeddings:
            model.lm_head.weight.copy_(coeff @ basis * 0.9 + 0.4 * torch.randn(vocab_size, d) / math.sqrt(d))
    model.eval()
    return model


def build_tiny_tokenizer(corpus_lines: Sequence[str], vocab_size: int = 1024, seed: int = 0):
    """Train a byte-level BPE dominated by English, mirroring the real problem.

    Real tokenizers are fit on corpora where English is over-represented; the
    consequence is that non-Latin scripts fall back toward byte level.  We
    reproduce that here by training on a corpus that is mostly the English
    lines, so the offline tests exhibit the same asymmetry the benchmark
    measures on production tokenizers — rather than a synthetic one.

    Falls back to :class:`~parity.tokenization.ByteTokenizer` if ``tokenizers``
    is unavailable.

    Claim: infrastructure, reduction — gives the offline suite a base tokenizer
    with genuine cross-script fertility asymmetry to reduce.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    except Exception:  # pragma: no cover - tokenizers is a transformers dep
        from parity.tokenization import ByteTokenizer

        return ByteTokenizer()

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tok.train_from_iterator(list(corpus_lines), trainer=trainer)

    class _TinyTok:
        name = "tiny-bpe"

        def __init__(self, t):
            self._t = t

        @property
        def vocab_size(self) -> int:
            """Id-space size.

            Claim: infrastructure.
            """
            return self._t.get_vocab_size()

        def encode(self, text: str) -> List[int]:
            """Encode to ids.

            Claim: infrastructure.
            """
            return self._t.encode(text).ids

        def decode(self, ids: Sequence[int]) -> str:
            """Decode ids to text.

            Claim: infrastructure.
            """
            return self._t.decode(list(ids))

        def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
            """Sub-word display strings.

            Claim: infrastructure.
            """
            return [self._t.id_to_token(int(i)) or "" for i in ids]

    return _TinyTok(tok)
