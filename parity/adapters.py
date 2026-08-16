"""Uniform access to a causal LM's embeddings and residual stream.

:mod:`parity.synthesis` and :mod:`parity.certificate` need four things from a
model and nothing else:

1. read an embedding row,
2. run a forward pass **from embeddings** (so a candidate row can be a free
   variable without ever being written into the weight matrix),
3. read the residual stream at every layer,
4. append rows to the input (and, if untied, output) embedding matrix.

Keeping that surface to four operations is what lets Parity target any
open-weight decoder — and it is also what makes the ``low-cost`` claim
checkable, because the FLOP accounting lives here, next to the forward pass it
counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

log = logging.getLogger("parity.adapters")


@dataclass
class Trace:
    """Result of one traced forward pass.

    ``hidden_states`` is ``(n_layers + 1)`` tensors of shape ``[B, T, d]``:
    index 0 is the embedding output, index ``l`` is the residual stream after
    block ``l``.  The synthesis objective is defined on these.

    Claim: infrastructure.
    """

    logits: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]
    n_tokens: int = 0


@runtime_checkable
class LMAdapter(Protocol):
    """The four-operation model interface Parity depends on.

    Claim: infrastructure.
    """

    d_model: int
    n_layers: int
    tied_embeddings: bool

    def vocab_size(self) -> int: ...
    def input_matrix(self) -> torch.Tensor: ...
    def output_matrix(self) -> Optional[torch.Tensor]: ...
    def trace(self, input_ids=None, inputs_embeds=None, attention_mask=None) -> Trace: ...
    def append_rows(self, input_rows: torch.Tensor, output_rows: Optional[torch.Tensor]) -> None: ...


class TorchLMAdapter:
    """Adapter for any module exposing the HF causal-LM call surface.

    Works unchanged for ``transformers`` models and for
    :class:`~parity.tiny.TinyCausalLM`.

    Claim: infrastructure, low-cost — also the single place forward-pass FLOPs
    are counted, so the cost comparison against continued pretraining is
    measured rather than asserted.
    """

    def __init__(self, model, name: str = "", pad_id: int = 0):
        self.model = model
        self.name = name or type(model).__name__
        self.pad_id = pad_id
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        emb = model.get_input_embeddings()
        self.d_model = int(emb.weight.shape[1])
        cfg = getattr(model, "config", None)
        self.n_layers = int(
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layers", None)
            or getattr(cfg, "n_layer", None)
            or len(getattr(model, "blocks", []))
        )
        out = model.get_output_embeddings()
        self.tied_embeddings = out is None or out.weight.data_ptr() == emb.weight.data_ptr()
        self.device = emb.weight.device
        self.dtype = emb.weight.dtype
        self._n_params = sum(p.numel() for p in model.parameters())
        self._forward_tokens = 0

    # -- basic accessors ----------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_id: str, dtype: str = "float32", device: str = "cpu", **kw) -> "TorchLMAdapter":
        """Load an open-weight model from the Hub or a local directory.

        ``float32`` by default: the synthesis step solves a least-squares
        problem whose conditioning is genuinely hurt by bf16 rounding, and the
        certificate compares KL divergences at the 1e-3 scale.  Cast to bf16
        *after* building a pack, not before.

        Claim: infrastructure, bound — numerical precision is a correctness
        issue for the guarantee, not a performance preference.
        """
        from transformers import AutoModelForCausalLM

        torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kw)
        model.to(device)
        return cls(model, name=model_id)

    def vocab_size(self) -> int:
        """Rows currently in the input embedding matrix.

        Claim: infrastructure.
        """
        return int(self.model.get_input_embeddings().weight.shape[0])

    def input_matrix(self) -> torch.Tensor:
        """The input embedding weight (live view, not a copy).

        Claim: infrastructure.
        """
        return self.model.get_input_embeddings().weight

    def output_matrix(self) -> Optional[torch.Tensor]:
        """The unembedding weight, or ``None`` when embeddings are tied.

        Claim: infrastructure.
        """
        if self.tied_embeddings:
            return None
        return self.model.get_output_embeddings().weight

    def unembed_matrix(self) -> torch.Tensor:
        """The matrix that maps a hidden state to logits, tied or not.

        Needed by the deterministic Lipschitz bound, which depends on its row
        norms.

        Claim: bound.
        """
        out = self.model.get_output_embeddings()
        return out.weight if out is not None else self.input_matrix()

    def embed_ids(self, ids: Sequence[int]) -> torch.Tensor:
        """Look up embedding rows for ``ids`` as ``[len(ids), d]``.

        Claim: infrastructure.
        """
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        return self.input_matrix().index_select(0, idx)

    # -- forward ------------------------------------------------------------

    def trace(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Trace:
        """One forward pass returning logits and every layer's residual stream.

        Gradients flow when ``inputs_embeds`` requires grad; all model
        parameters are frozen in ``__init__``, so the only thing a backward pass
        can touch is the candidate embedding.  That is the mechanical reason
        Parity is not continued pretraining: there is no path from the loss to a
        weight.

        Claim: non-regression, low-cost — the frozen-parameter invariant is
        enforced here, not merely promised in the README.
        """
        kwargs = {"output_hidden_states": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if inputs_embeds is not None:
            out = self.model(inputs_embeds=inputs_embeds, **kwargs)
            n_tok = int(inputs_embeds.shape[0] * inputs_embeds.shape[1])
        else:
            ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
            if ids.dim() == 1:
                ids = ids[None, :]
            out = self.model(input_ids=ids, **kwargs)
            n_tok = int(ids.numel())
        self._forward_tokens += n_tok
        hs = out.hidden_states
        return Trace(logits=out.logits, hidden_states=tuple(hs), n_tokens=n_tok)

    def embeds_for(self, id_rows: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-pad ragged id rows into ``(inputs_embeds, attention_mask)``.

        Right padding is safe under causal attention: a padded position can only
        attend to the left, and nothing to its left ever attends to it, so the
        hidden states at real positions are unchanged.  The mask is still passed
        so that models which use it for anything else behave correctly.

        Claim: infrastructure.
        """
        B = len(id_rows)
        T = max(len(r) for r in id_rows)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, T), dtype=torch.long, device=self.device)
        for i, row in enumerate(id_rows):
            ids[i, : len(row)] = torch.as_tensor(list(row), dtype=torch.long, device=self.device)
            mask[i, : len(row)] = 1
        return self.input_matrix()[ids], mask

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        """Serving-path forward with an explicit KV cache and positions.

        Explicit ``position_ids`` are what make **left** padding correct, which
        is what lets requests using different tokenizer views share one batch.
        Batch sharing is the reason multi-view serving costs almost nothing —
        see :mod:`parity.serving.multi_tokenizer`.

        Claim: low-cost, non-regression.
        """
        kwargs = {"use_cache": use_cache}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        out = self.model(input_ids=input_ids, **kwargs)
        self._forward_tokens += int(input_ids.numel())
        return out.logits, getattr(out, "past_key_values", None)

    # -- mutation -----------------------------------------------------------

    def append_rows(self, input_rows: torch.Tensor, output_rows: Optional[torch.Tensor] = None) -> Tuple[int, int]:
        """Append embedding rows, leaving every existing row untouched.

        Returns the ``[start, end)`` id range assigned.  Asserts afterwards that
        the original rows are bit-identical — the English non-regression claim
        rests on that and it costs one comparison to verify.

        Claim: non-regression — the exactness of "English is unaffected" is
        established here.
        """
        emb = self.model.get_input_embeddings()
        old_v, d = emb.weight.shape
        n_new = int(input_rows.shape[0])
        if input_rows.shape[1] != d:
            raise ValueError(f"row width {input_rows.shape[1]} != d_model {d}")
        if not self.tied_embeddings and output_rows is None:
            raise ValueError("model has untied embeddings; output_rows is required")

        before = emb.weight.detach().clone()
        new_size = old_v + n_new
        resize = getattr(self.model, "resize_token_embeddings", None)
        if resize is not None:
            resize(new_size)
        else:
            self._manual_resize(new_size)

        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            emb.weight[old_v:new_size] = input_rows.to(emb.weight.dtype).to(emb.weight.device)
            out = self.model.get_output_embeddings()
            if out is not None and not self.tied_embeddings:
                out.weight[old_v:new_size] = output_rows.to(out.weight.dtype).to(out.weight.device)
            elif out is not None and self.tied_embeddings:
                # Tied: writing the input rows already set the output rows, but
                # some resize implementations un-tie. Re-tie explicitly.
                out.weight = emb.weight

        after = self.model.get_input_embeddings().weight[:old_v].detach()
        if not torch.equal(before.to(after.dtype), after):
            raise AssertionError(
                "append_rows modified pre-existing embedding rows; the base-view "
                "non-regression guarantee would not hold"
            )
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "vocab_size"):
            cfg.vocab_size = new_size
        log.info("appended %d rows: ids [%d, %d)", n_new, old_v, new_size)
        return old_v, new_size

    def _manual_resize(self, new_size: int) -> None:
        import torch.nn as nn

        emb = self.model.get_input_embeddings()
        old_v, d = emb.weight.shape
        new_emb = nn.Embedding(new_size, d, device=emb.weight.device, dtype=emb.weight.dtype)
        with torch.no_grad():
            new_emb.weight[:old_v] = emb.weight
            new_emb.weight[old_v:] = 0
        self.model.embed_tokens = new_emb
        out = self.model.get_output_embeddings()
        if out is not None:
            if self.tied_embeddings:
                new_head = nn.Linear(d, new_size, bias=False, device=out.weight.device, dtype=out.weight.dtype)
                new_head.weight = new_emb.weight
            else:
                new_head = nn.Linear(d, new_size, bias=False, device=out.weight.device, dtype=out.weight.dtype)
                with torch.no_grad():
                    new_head.weight[:old_v] = out.weight
                    new_head.weight[old_v:] = 0
            self.model.lm_head = new_head
        for p in self.model.parameters():
            p.requires_grad_(False)

    # -- cost ---------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total parameter count, for FLOP accounting.

        Claim: low-cost.
        """
        return self._n_params

    @property
    def forward_tokens(self) -> int:
        """Tokens pushed through this adapter since construction.

        Claim: low-cost — the measured basis of the cost comparison in
        benchmark metric (4).
        """
        return self._forward_tokens

    def reset_cost(self) -> None:
        """Zero the token counter before a stage you want to price separately.

        Claim: low-cost.
        """
        self._forward_tokens = 0

    def flops(self, tokens: Optional[int] = None, backward: bool = False) -> float:
        """FLOPs for ``tokens`` at this model's size.

        Uses the standard accounting: ``2 * N`` FLOPs per token for a forward
        pass, ``6 * N`` for forward+backward (Kaplan et al.).  Parity's synthesis
        step backpropagates only to a ``d``-dimensional input, but the backward
        pass through the frozen network still costs the same as a training
        backward pass, so we charge for it honestly.

        Claim: low-cost — an under-counted denominator would fake the headline
        "orders of magnitude cheaper" result, so the accounting is deliberately
        unfavourable to us.
        """
        t = self._forward_tokens if tokens is None else tokens
        return float(t) * self._n_params * (6.0 if backward else 2.0)

    def embedding_param_cost(self, n_new: int) -> int:
        """Parameters added by ``n_new`` Parity tokens.

        Claim: low-cost — the entire storage footprint of a pack, and the
        quantity the vocabulary budget constrains.
        """
        return n_new * self.d_model * (1 if self.tied_embeddings else 2)
