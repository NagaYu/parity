"""EmbeddingSynthesis — build embedding rows for new tokens, without training.

This is the core of Parity.  For a new token ``t`` whose expansion is the base
sequence ``(v_1, …, v_k)``, we need an input embedding such that feeding the
single token ``t`` leaves the network in the same state that feeding
``v_1 … v_k`` would have.

Stage (i): composition baseline
-------------------------------
``e_0 = norm_match(Σ_i w_i · E[v_i])``.  Weights default to inverse-frequency
(rare sub-tokens carry more of the unit's identity — the SIF weighting), and the
result is rescaled to the norm typical of the embedding matrix.  The rescaling
is not cosmetic: averaging *k* nearly-orthogonal vectors shrinks the norm by
about ``1/sqrt(k)``, which moves the vector off the manifold the model's first
RMSNorm was calibrated on.  Plain mean-of-sub-tokens (no rescaling) is exactly
the published zero-shot transfer baseline and is available as
``composition="mean"``; that is benchmark condition (C).

Stage (ii): subspace Gauss-Newton least squares
-----------------------------------------------
Take a handful of calibration contexts in which the expansion actually occurs.
Run the model twice — once on ``prefix + v_1..v_k + suffix`` and once on
``prefix + t + suffix`` — and align the residual streams position-for-position
over the shared suffix, plus the "unit readout" position (the state right after
the unit has been consumed).  Minimise

    L(e) = Σ_c Σ_ℓ Σ_p  ‖ h_new[ℓ,p] − h_orig[ℓ,p] ‖² / σ_ℓ²   +   λ‖e − e_0‖²

restricting ``e = e_0 + B z`` with ``B ∈ R^{d×q}``, ``q ≈ 8``, spanned by the
sub-token embeddings, their mean, and top principal directions of the embedding
matrix.  In that subspace the Gauss-Newton normal equations are ``q × q`` and
solved in closed form; the Jacobian is obtained by ``q`` finite-difference
forward passes, which are **shared across every candidate in a chunk** because
each calibration sequence contains exactly one candidate.  That vectorisation is
what makes the cost claim hold: the number of forward passes is a function of
``q`` and the iteration count, not of the number of tokens being synthesised.

Why this is not continued pretraining
-------------------------------------
Every model parameter is frozen in :class:`~parity.adapters.TorchLMAdapter`,
including through the backward path.  The only free variables are ``q`` numbers
per new token.  There is no language-modelling loss, no corpus pass, and no
optimiser state over weights.  A build touches a few thousand calibration
tokens; continued pretraining touches billions.  Benchmark metric (4) measures
that gap rather than asserting it.

An honest caveat
----------------
Merging shifts every token after the unit one or more positions to the left, so
the new sequence is evaluated at different absolute positions than the old one.
With RoPE this is mostly relative and therefore mild, but it puts a non-zero
floor under the residual: perfect matching is not achievable, and the
certificate in :mod:`parity.certificate` is what turns that residual floor into
a number a deployer can accept or refuse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from parity.adapters import TorchLMAdapter
from parity.types import MergeCandidate, SynthesizedToken

log = logging.getLogger("parity.synthesis")


# ---------------------------------------------------------------------------
# Calibration contexts
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """One calibration occurrence of a candidate inside a real sentence.

    Claim: non-regression — synthesis and certification are both defined
    relative to contexts the expansion actually appears in, not synthetic ones.
    """

    prefix: Tuple[int, ...]
    expansion: Tuple[int, ...]
    suffix: Tuple[int, ...]

    @property
    def orig_ids(self) -> List[int]:
        """Token ids of the unmerged sequence.

        Claim: infrastructure.
        """
        return list(self.prefix) + list(self.expansion) + list(self.suffix)

    def new_ids(self, new_token_placeholder: int) -> List[int]:
        """Token ids of the merged sequence (the merged slot is a placeholder).

        Claim: infrastructure.
        """
        return list(self.prefix) + [new_token_placeholder] + list(self.suffix)


@dataclass
class ChunkTargets:
    """Everything one batched forward over the original expansions yields.

    ``states`` are the residual-stream vectors the merged sequence must
    reproduce; ``emit_*`` are the ingredients of the emission target (the logit
    the new token needs so it is generated with the right probability).

    Claim: non-regression, low-cost — one pass, both objectives.
    """

    states: Any  # [N, L, d]
    weights: Any  # [N]
    owners: Any  # [N] long, index within the chunk
    emit_h: Any  # [M, d]
    emit_y: Any  # [M]
    emit_owner: Any  # [M] long


class CalibrationIndex:
    """All occurrences of all candidates in a corpus, found in one scan.

    Built as a trie over candidate id-sequences; at each corpus position we walk
    the trie and record *every* terminal on the path, so overlapping and nested
    candidates are all indexed.  One pass over the corpus serves thousands of
    candidates.

    Claim: low-cost — turns "find contexts for N candidates" from ``O(N·corpus)``
    into ``O(corpus·max_len)``.
    """

    def __init__(self, candidates: Sequence[MergeCandidate], prefix_tokens: int = 12, suffix_tokens: int = 6):
        from parity.tokenization import MergeTrie

        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self._trie = MergeTrie()
        self._by_key: Dict[str, MergeCandidate] = {}
        for idx, c in enumerate(candidates):
            self._trie.add(c.ids, idx)
            self._by_key[c.key] = c
        self._candidates = list(candidates)
        self._index_of_key: Dict[str, int] = {c.key: i for i, c in enumerate(candidates)}
        self._contexts: Dict[int, List[Context]] = {i: [] for i in range(len(candidates))}

    def scan(self, docs: Sequence[Sequence[int]], max_per_candidate: int = 8) -> None:
        """Populate contexts from tokenized documents.

        Claim: low-cost.
        """
        for doc in docs:
            n = len(doc)
            for i in range(n):
                node = self._trie
                j = i
                while j < n:
                    child = node.children.get(doc[j])
                    if child is None:
                        break
                    node = child
                    j += 1
                    if node.token is None:
                        continue
                    idx = node.token
                    bucket = self._contexts[idx]
                    if len(bucket) >= max_per_candidate:
                        continue
                    bucket.append(
                        Context(
                            prefix=tuple(doc[max(0, i - self.prefix_tokens) : i]),
                            expansion=tuple(doc[i:j]),
                            suffix=tuple(doc[j : j + self.suffix_tokens]),
                        )
                    )

    def contexts(self, candidate_index: int) -> List[Context]:
        """Contexts collected for one candidate (possibly empty).

        Claim: infrastructure.
        """
        return self._contexts.get(candidate_index, [])

    def contexts_for(self, candidate: MergeCandidate) -> List[Context]:
        """Contexts for a candidate looked up by identity, not by position.

        Certification uses a *different* index built on a held-out corpus slice,
        so it must be able to find a candidate without knowing its position in
        the mining order.

        Claim: bound — a bound measured on the data the embedding was fitted to
        is not a bound; this is the lookup that keeps the two sets disjoint.
        """
        idx = self._index_of_key.get(candidate.key)
        return [] if idx is None else self._contexts.get(idx, [])

    def coverage(self) -> float:
        """Fraction of candidates that got at least one context.

        A candidate with no context cannot be certified and is dropped; this
        number tells the operator whether their calibration corpus is too small.

        Claim: bound — no context means no evidence means no certificate.
        """
        if not self._candidates:
            return 0.0
        return sum(1 for v in self._contexts.values() if v) / len(self._candidates)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SynthesisConfig:
    """Knobs for :class:`EmbeddingSynthesizer`.

    Defaults are chosen so that a build over a few thousand candidates on a
    0.5B model costs on the order of ``10^16`` FLOPs — two to three orders of
    magnitude below a continued-pretraining run on the same model.  Raising
    ``subspace_dim``, ``gn_iters`` or ``max_contexts`` improves the fit and
    costs linearly more; the benchmark sweeps them.

    Claim: low-cost, non-regression — every field here is a point on the
    cost/fidelity trade-off the Pareto figure plots.
    """

    composition: str = "inverse_frequency"  # mean | sum | first | inverse_frequency
    norm_match: bool = True
    solver: str = "gn"  # composition | gn | gn+adam
    subspace_dim: int = 8
    gn_iters: int = 2
    damping: float = 1e-2
    ridge: float = 1e-3  # pull toward the composition init
    fd_eps: float = 5e-3  # relative finite-difference step
    max_contexts: int = 8
    prefix_tokens: int = 12
    suffix_tokens: int = 6
    n_layers_probed: int = 3
    readout_weight: float = 2.0
    chunk_size: int = 32
    adam_steps: int = 0
    adam_lr: float = 5e-3
    # Output-embedding ridge regression.
    fit_output_embedding: bool = True
    output_ridge: float = 1e-1
    #: Largest allowed ``‖w − anchor‖ / ‖anchor‖`` for a fitted output row.
    #:
    #: Without this, a row can satisfy its emission targets on the handful of
    #: calibration contexts by growing a large component that fires on unrelated
    #: text — stealing probability mass from ordinary tokens, including in other
    #: languages.  :class:`~parity.certificate.NegativeProbe` measures that
    #: damage; this keeps the solve inside the region where it does not happen.
    output_trust_region: float = 1.0
    #: Contexts used for the output-row fit.  Larger than ``max_contexts``
    #: because that fit is one closed-form solve on states we already computed,
    #: so extra contexts cost nothing beyond a slightly wider batch.
    output_contexts: int = 24
    #: Weight of the emission term inside the *input* least-squares objective.
    #: Only used when the model ties its embeddings, where the input row is also
    #: the output row and there is no second vector to fit.
    emit_weight: float = 0.5
    seed: int = 0


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class EmbeddingSynthesizer:
    """Turn merge candidates into embedding rows for a frozen model.

    Claim: non-regression, low-cost — produces rows whose behaviour matches the
    original expansion, at a cost that is a rounding error next to training.
    """

    def __init__(
        self,
        adapter: TorchLMAdapter,
        config: Optional[SynthesisConfig] = None,
        unigram_freq: Optional[Dict[int, int]] = None,
    ):
        self.lm = adapter
        self.cfg = config or SynthesisConfig()
        self.unigram_freq = unigram_freq or {}
        self._pca: Optional[torch.Tensor] = None
        self._emb_norm: Optional[float] = None
        self._probe_layers = self._pick_layers()
        torch.manual_seed(self.cfg.seed)

    # -- layer selection ----------------------------------------------------

    def _pick_layers(self) -> List[int]:
        n = self.lm.n_layers
        k = max(1, min(self.cfg.n_layers_probed, n))
        if k == 1:
            return [n]
        # Evenly spaced, always including the final residual stream.
        return sorted({int(round(n * (i + 1) / k)) for i in range(k)})

    @property
    def probe_layers(self) -> List[int]:
        """Which residual-stream layers the objective matches.

        Includes the final layer always: that is the one the unembedding reads,
        so it is the layer whose mismatch turns directly into logit drift.

        Claim: bound — ties the synthesis objective to the quantity the
        certificate measures.
        """
        return list(self._probe_layers)

    # -- stage (i): composition --------------------------------------------

    def _mean_embedding_norm(self) -> float:
        if self._emb_norm is None:
            W = self.lm.input_matrix()
            n = min(20000, W.shape[0])
            self._emb_norm = float(W[:n].float().norm(dim=1).mean())
        return self._emb_norm

    def compose(self, ids: Sequence[int]) -> torch.Tensor:
        """Stage (i): a weighted combination of the sub-token embeddings.

        ``composition="mean"`` with ``norm_match=False`` reproduces the standard
        zero-shot vocabulary-transfer baseline (benchmark condition C), so the
        two conditions differ in exactly the intended way.

        Claim: non-regression, low-cost — a usable embedding for zero forward
        passes; the honest floor the least-squares stage must beat.
        """
        return self.compose_from(self.lm.input_matrix(), ids)

    def compose_from(self, matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
        """Apply the composition rule to an arbitrary row matrix.

        Used for the *output* side of the zero-shot transfer baseline: with
        untied embeddings that baseline still needs an unembedding row, and the
        published recipe takes the same mean over the sub-tokens' output rows.
        Reusing one implementation keeps the baseline faithful instead of
        accidentally favourable.

        Claim: non-regression — a baseline implemented differently from its
        published form is not a baseline.
        """
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=matrix.device)
        E = matrix.index_select(0, idx).float()  # [k, d]
        k = E.shape[0]
        mode = self.cfg.composition
        if mode == "first":
            w = torch.zeros(k)
            w[0] = 1.0
        elif mode == "sum":
            w = torch.ones(k)
        elif mode == "inverse_frequency":
            # SIF-style: a sub-token that is everywhere carries little of the
            # unit's identity, so down-weight it.
            a = 1e-4
            total = max(1, sum(self.unigram_freq.values())) if self.unigram_freq else 1
            probs = torch.tensor([(self.unigram_freq.get(int(i), 1) / total) if self.unigram_freq else 1.0 for i in ids])
            w = a / (a + probs) if self.unigram_freq else torch.ones(k)
            w = w / w.sum().clamp_min(1e-9)
        else:  # "mean"
            w = torch.ones(k) / k
        w = w.to(E.device).to(E.dtype)
        e = (w[:, None] * E).sum(0)
        if self.cfg.norm_match:
            target = float(matrix[: min(20000, matrix.shape[0])].float().norm(dim=1).mean())
            e = e * (target / float(e.norm().clamp_min(1e-9)))
        return e

    # -- basis --------------------------------------------------------------

    def _pca_directions(self, q: int) -> torch.Tensor:
        if self._pca is None or self._pca.shape[1] < q:
            W = self.lm.input_matrix()
            n = min(8192, W.shape[0])
            X = W[:n].float()
            X = X - X.mean(0, keepdim=True)
            try:
                _, _, V = torch.pca_lowrank(X, q=min(max(q, 8), min(X.shape) - 1))
            except Exception:  # pragma: no cover - tiny matrices
                V = torch.randn(X.shape[1], q)
                V, _ = torch.linalg.qr(V)
            self._pca = V
        return self._pca[:, :q]

    def _basis(self, ids: Sequence[int], q: int) -> torch.Tensor:
        """Orthonormal ``[d, q]`` search subspace for one candidate.

        Spanned by the sub-token embeddings (the directions most likely to
        matter), their mean, and top principal directions of the embedding
        matrix (which keep the solution on the manifold the model expects).

        Claim: low-cost — restricting to ``q`` directions is what reduces the
        Jacobian to ``q`` forward passes instead of ``d``.
        """
        d = self.lm.d_model
        cols = [self.lm.embed_ids(ids).float()]  # [k, d]
        cols.append(cols[0].mean(0, keepdim=True))
        cols.append(self._pca_directions(min(q, d)).T)  # [q, d]
        cols.append(torch.randn(q, d) / (d ** 0.5))
        M = torch.cat(cols, dim=0).T  # [d, *]
        Qm, _ = torch.linalg.qr(M)
        return Qm[:, :q].contiguous()

    # -- residual objective -------------------------------------------------

    def _aligned_positions(self, ctx: Context) -> Tuple[List[int], List[int], List[float]]:
        P, k = len(ctx.prefix), len(ctx.expansion)
        orig, new, w = [P + k - 1], [P], [self.cfg.readout_weight]
        for j in range(len(ctx.suffix)):
            orig.append(P + k + j)
            new.append(P + 1 + j)
            w.append(1.0)
        return orig, new, w

    @torch.no_grad()
    def _target_states(self, contexts: Sequence[Tuple[int, Context]], cand_pos: Dict[int, int]) -> "ChunkTargets":
        """Run the *original* expansions once; gather state and emission targets.

        One batched forward pass produces everything the chunk needs:

        * the residual-stream states the merged sequence must reproduce, and
        * the emission targets ``y_c`` — the logit the new token must receive so
          that ``P_new(t | c)`` equals the original model's probability for the
          whole expansion (derivation in :meth:`_fit_output_rows`).

        Computing both here rather than in two passes halves the dominant cost
        of the synthesis stage.

        Claim: non-regression, low-cost.
        """
        rows, weights, owners = [], [], []
        emit_h, emit_y, emit_owner = [], [], []
        id_rows = [ctx.orig_ids for _, ctx in contexts]
        embeds, mask = self.lm.embeds_for(id_rows)
        tr = self.lm.trace(inputs_embeds=embeds, attention_mask=mask)
        H = torch.stack([tr.hidden_states[l] for l in self._probe_layers], dim=2)  # [B, T, L, d]
        last = tr.hidden_states[-1]
        for bi, (cand_idx, ctx) in enumerate(contexts):
            o_pos, _, w = self._aligned_positions(ctx)
            for p, wt in zip(o_pos, w):
                rows.append(H[bi, p])
                weights.append(wt)
                owners.append(cand_pos[cand_idx])
            P, k = len(ctx.prefix), len(ctx.expansion)
            if P == 0:
                continue  # nothing predicts the first sub-token, so no target
            logits = tr.logits[bi].float()
            s = torch.zeros((), dtype=torch.float32)
            for i in range(k):
                s = s + torch.log_softmax(logits[P - 1 + i], dim=-1)[ctx.expansion[i]]
            s = s.clamp(max=-1e-4)
            logZ = torch.logsumexp(logits[P - 1], dim=-1)
            emit_h.append(last[bi, P - 1].float())
            emit_y.append(s + logZ - torch.log1p(-torch.exp(s)))
            emit_owner.append(cand_pos[cand_idx])
        d = self.lm.d_model
        return ChunkTargets(
            states=torch.stack(rows).float(),
            weights=torch.tensor(weights, dtype=torch.float32),
            owners=torch.tensor(owners, dtype=torch.long),
            emit_h=torch.stack(emit_h).float() if emit_h else torch.zeros(0, d),
            emit_y=torch.stack(emit_y).float() if emit_y else torch.zeros(0),
            emit_owner=torch.tensor(emit_owner, dtype=torch.long) if emit_owner else torch.zeros(0, dtype=torch.long),
        )

    def _select_state_rows(
        self, flat: Sequence[Tuple[int, Context]], is_state: Sequence[bool], tg: "ChunkTargets"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep only the state-matching rows belonging to state-fitting contexts.

        :meth:`_target_states` emits ``1 + len(suffix)`` rows per context, so the
        mapping from contexts to rows is ragged; this walks it once.

        Claim: low-cost — lets extra contexts feed the free closed-form output
        fit without inflating the iterative solve.
        """
        keep: List[int] = []
        row = 0
        for (_, ctx), want in zip(flat, is_state):
            n = 1 + len(ctx.suffix)
            if want:
                keep.extend(range(row, row + n))
            row += n
        idx = torch.tensor(keep, dtype=torch.long)
        return tg.states.index_select(0, idx), tg.weights.index_select(0, idx), tg.owners.index_select(0, idx)

    @torch.no_grad()
    def _current_states(
        self, contexts: Sequence[Tuple[int, Context]], embeddings: torch.Tensor, cand_pos: Dict[int, int]
    ) -> torch.Tensor:
        """Run the *merged* sequences with a candidate row spliced in.

        The candidate embedding is never written into the weight matrix; it is
        placed directly into ``inputs_embeds``.  That is what allows thousands
        of candidates to be evaluated against one frozen model.

        Claim: low-cost, non-regression.
        """
        id_rows = [ctx.new_ids(0) for _, ctx in contexts]
        embeds, mask = self.lm.embeds_for(id_rows)
        embeds = embeds.clone()
        for bi, (cand_idx, ctx) in enumerate(contexts):
            embeds[bi, len(ctx.prefix)] = embeddings[cand_pos[cand_idx]].to(embeds.dtype)
        tr = self.lm.trace(inputs_embeds=embeds, attention_mask=mask)
        H = torch.stack([tr.hidden_states[l] for l in self._probe_layers], dim=2)
        rows = []
        for bi, (_, ctx) in enumerate(contexts):
            _, n_pos, _ = self._aligned_positions(ctx)
            for p in n_pos:
                rows.append(H[bi, p])
        return torch.stack(rows).float()

    # -- main entry point ---------------------------------------------------

    def synthesize(
        self,
        candidates: Sequence[MergeCandidate],
        index: CalibrationIndex,
    ) -> List[SynthesizedToken]:
        """Synthesise embedding rows for every candidate that has calibration data.

        Candidates with no observed context are skipped: an embedding fitted on
        nothing cannot be certified, and shipping it would be exactly the kind
        of unbacked claim this project exists to avoid.

        Claim: non-regression, low-cost — the headline method.
        """
        out: List[SynthesizedToken] = []
        cfg = self.cfg
        usable = [(i, c) for i, c in enumerate(candidates) if index.contexts(i)]
        if len(usable) < len(candidates):
            log.info("skipping %d candidates with no calibration context", len(candidates) - len(usable))

        for start in range(0, len(usable), cfg.chunk_size):
            chunk = usable[start : start + cfg.chunk_size]
            out.extend(self._synthesize_chunk(chunk, index))
            log.info("synthesised %d/%d candidates", min(start + cfg.chunk_size, len(usable)), len(usable))
        return out

    def _synthesize_chunk(
        self, chunk: Sequence[Tuple[int, MergeCandidate]], index: CalibrationIndex
    ) -> List[SynthesizedToken]:
        cfg = self.cfg
        t0 = time.time()
        self.lm.reset_cost()

        cand_pos = {cand_idx: pos for pos, (cand_idx, _) in enumerate(chunk)}
        C = len(chunk)
        d = self.lm.d_model

        e0 = torch.stack([self.compose(c.ids) for _, c in chunk])  # [C, d]

        n_ctx = max(cfg.max_contexts, cfg.output_contexts if cfg.fit_output_embedding else 0)
        flat: List[Tuple[int, Context]] = []
        is_state: List[bool] = []
        state_ctx: List[Tuple[int, Context]] = []
        for cand_idx, _ in chunk:
            for i, ctx in enumerate(index.contexts(cand_idx)[:n_ctx]):
                flat.append((cand_idx, ctx))
                keep = i < cfg.max_contexts
                is_state.append(keep)
                if keep:
                    state_ctx.append((cand_idx, ctx))
        if not flat:
            return []

        tg = self._target_states(flat, cand_pos)
        # Only the first `max_contexts` contexts per candidate drive the (costly)
        # iterative state-matching solve; the rest ride along on the single
        # original forward pass and feed the closed-form output fit.
        state_targets, state_weights, state_owner = self._select_state_rows(flat, is_state, tg)
        # Per-layer scale so deep layers (large norm) do not dominate shallow ones.
        scale = state_targets.float().pow(2).mean(dim=(0, 2)).sqrt().clamp_min(1e-6)  # [L]
        m = state_targets.shape[1] * state_targets.shape[2]

        tied = self.lm.tied_embeddings
        emit_scale = float(tg.emit_y.abs().mean().clamp_min(1.0)) if tg.emit_y.numel() else 1.0

        def residual(emb: torch.Tensor) -> torch.Tensor:
            cur = self._current_states(state_ctx, emb, cand_pos)
            diff = (cur - state_targets) / scale[None, :, None]
            r = diff.reshape(diff.shape[0], -1) * state_weights[:, None].sqrt()
            if tied and cfg.emit_weight > 0 and tg.emit_y.numel():
                # With tied embeddings the input row *is* the output row, so the
                # emission target has to be part of this objective — there is no
                # second vector to fit it with.
                pred = (tg.emit_h * emb[tg.emit_owner]).sum(-1)
                e_res = (pred - tg.emit_y) / emit_scale * (cfg.emit_weight ** 0.5)
                block = torch.zeros(e_res.shape[0], r.shape[1], dtype=r.dtype)
                block[:, 0] = e_res
                r = torch.cat([r, block], dim=0)
            return r

        owner_t = state_owner
        if tied and cfg.emit_weight > 0 and tg.emit_y.numel():
            owner_t = torch.cat([state_owner, tg.emit_owner])

        r0 = residual(e0)
        loss0 = _per_owner_sumsq(r0, owner_t, C)

        if cfg.solver == "composition":
            emb = e0
            lossN = loss0
            solver_name = "composition"
        else:
            emb, lossN = self._gauss_newton(e0, residual, owner_t, C, chunk)
            solver_name = cfg.solver
            if cfg.adam_steps > 0 and cfg.solver.endswith("adam"):
                emb, lossN = self._adam_refine(
                    emb, state_ctx, cand_pos, state_targets, state_weights, state_owner, C, scale
                )

        out_rows = None
        if not tied:
            if cfg.fit_output_embedding:
                out_rows = self._fit_output_rows(chunk, tg)
            else:
                # An untied model still needs an unembedding row. The zero-shot
                # transfer baseline composes one from the sub-tokens' output
                # rows, exactly as it composes the input row.
                W = self.lm.unembed_matrix()
                out_rows = torch.stack([self.compose_from(W, c.ids) for _, c in chunk])

        seconds = time.time() - t0
        flops = self.lm.flops(backward=(cfg.adam_steps > 0))
        results: List[SynthesizedToken] = []
        for pos, (cand_idx, cand) in enumerate(chunk):
            results.append(
                SynthesizedToken(
                    candidate=cand,
                    input_embedding=emb[pos].detach().clone(),
                    output_embedding=None if out_rows is None else out_rows[pos].detach().clone(),
                    solver=solver_name,
                    residual_before=float(loss0[pos]),
                    residual_after=float(lossN[pos]),
                    n_calibration=len(index.contexts(cand_idx)[: cfg.max_contexts]),
                    seconds=seconds / max(1, len(chunk)),
                    flops=flops / max(1, len(chunk)),
                )
            )
        return results

    # -- stage (ii): Gauss-Newton ------------------------------------------

    def _gauss_newton(self, e0, residual, owner_t, C: int, chunk) -> Tuple[torch.Tensor, torch.Tensor]:
        """Damped Gauss-Newton in a ``q``-dimensional subspace, per candidate.

        The Jacobian is built by finite differences along the ``q`` basis
        directions.  Because every calibration sequence contains exactly one
        candidate, all ``C`` candidates in a chunk are perturbed along their own
        ``j``-th basis vector in the *same* forward pass, so the whole chunk
        costs ``(q+1) · gn_iters`` forwards regardless of ``C``.

        Levenberg-Marquardt damping plus an explicit accept/reject step means a
        candidate whose linearisation is poor keeps its composition embedding
        rather than being made worse — the objective is monotone by
        construction.

        Claim: non-regression, low-cost — the least-squares stage required by
        the method, at ``O(q)`` forward passes for an entire chunk.
        """
        cfg = self.cfg
        q = min(cfg.subspace_dim, self.lm.d_model)
        B = torch.stack([self._basis(c.ids, q) for _, c in chunk])  # [C, d, q]
        emb = e0.clone()
        r = residual(emb)
        loss = _per_owner_sumsq(r, owner_t, C)
        eps = cfg.fd_eps * self._mean_embedding_norm()

        for it in range(cfg.gn_iters):
            cols = []
            for j in range(q):
                pert = emb + eps * B[:, :, j]
                rj = residual(pert)
                cols.append((rj - r) / eps)
            J = torch.stack(cols, dim=-1)  # [N, m, q]

            new_emb = emb.clone()
            for c in range(C):
                sel = owner_t == c
                if not bool(sel.any()):
                    continue
                Jc = J[sel].reshape(-1, q)
                rc = r[sel].reshape(-1)
                A = Jc.T @ Jc
                g = Jc.T @ rc
                # Ridge toward the composition init, plus LM damping.
                lam = cfg.damping * float(torch.diagonal(A).mean().clamp_min(1e-8))
                A = A + (lam + cfg.ridge) * torch.eye(q)
                try:
                    dz = torch.linalg.solve(A, -g)
                except Exception:  # pragma: no cover
                    dz = -torch.linalg.lstsq(A, g[:, None]).solution[:, 0]
                new_emb[c] = emb[c] + B[c] @ dz

            r_new = residual(new_emb)
            loss_new = _per_owner_sumsq(r_new, owner_t, C)
            improved = loss_new < loss
            # Accept only candidates that actually improved; the rest keep the
            # previous iterate, so the objective never increases for any token.
            emb = torch.where(improved[:, None], new_emb, emb)
            r = torch.where(improved[owner_t][:, None], r_new, r)
            loss = torch.minimum(loss, loss_new)
            log.debug("GN iter %d: improved %d/%d, mean loss %.5f", it, int(improved.sum()), C, float(loss.mean()))
        return emb, loss

    def _adam_refine(self, emb, flat, cand_pos, targets, weights, owner_t, C, scale):
        """Optional full-``d`` refinement of the same least-squares objective.

        Off by default.  It costs a backward pass per step and buys little once
        the subspace solve has converged, but it is available for operators who
        want the last few percent of residual and can pay for it.

        Claim: non-regression, low-cost — an explicit knob on the fidelity/cost
        trade-off rather than a hidden default.
        """
        cfg = self.cfg
        z = emb.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.adam_lr * self._mean_embedding_norm())
        for _ in range(cfg.adam_steps):
            opt.zero_grad(set_to_none=True)
            id_rows = [ctx.new_ids(0) for _, ctx in flat]
            embeds, mask = self.lm.embeds_for(id_rows)
            embeds = embeds.clone()
            for bi, (cand_idx, ctx) in enumerate(flat):
                embeds[bi, len(ctx.prefix)] = z[cand_pos[cand_idx]].to(embeds.dtype)
            tr = self.lm.trace(inputs_embeds=embeds, attention_mask=mask)
            H = torch.stack([tr.hidden_states[l] for l in self._probe_layers], dim=2)
            rows = []
            for bi, (_, ctx) in enumerate(flat):
                _, n_pos, _ = self._aligned_positions(ctx)
                for p in n_pos:
                    rows.append(H[bi, p])
            cur = torch.stack(rows).float()
            diff = (cur - targets) / scale[None, :, None]
            loss = ((diff.reshape(diff.shape[0], -1) ** 2).sum(-1) * weights).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = z.detach()
        return final, _per_owner_sumsq(
            (self._current_states(flat, final, cand_pos) - targets).reshape(len(targets), -1)
            * weights[:, None].sqrt(),
            owner_t,
            C,
        )

    # -- output embeddings --------------------------------------------------

    @torch.no_grad()
    def _fit_output_rows(self, chunk, tg: "ChunkTargets") -> torch.Tensor:
        """Closed-form ridge regression for the unembedding row of each token.

        We want the augmented model to emit the new token with the probability
        the original model assigned to its *whole expansion*:

            P_new(t | c) = Π_i P_orig(v_i | c, v_<i) =: exp(s_c).

        Since the base rows are untouched, the softmax denominator is
        ``Z_base(c) + exp(z_t)``, and solving for ``z_t`` gives the target
        exactly:

            y_c = s_c + log Z_base(c) − log(1 − exp(s_c)).

        Rather than regress ``y`` directly — it has magnitude ~|log Z| and is
        badly conditioned with a handful of samples — we **anchor at the first
        sub-token's output row** and fit only the residual:

            w = W_U[v_1] + argmin_δ ‖Hδ − ρ‖² + λ‖δ‖²,   ρ_c = y_c − W_U[v_1]·h_c.

        The anchor is principled: ``y_c = z_{v_1}(c) + Σ_{i≥2} log P(v_i | ·) −
        log(1 − e^{s_c})``, so ``ρ_c`` is (minus) the surprisal of the *rest* of
        the expansion given its first sub-token.  For a good merge candidate the
        continuation is nearly deterministic, so ``ρ`` is small and nearly
        constant — which is why the fit is easy exactly when the candidate is
        worth adopting.  A large residual here is a signal that the candidate is
        not really one unit, and the certificate rejects it downstream.

        Claim: non-regression — makes the new token *emittable* with the right
        mass, so generation is preserved and not only prompt cost.
        """
        d = self.lm.d_model
        W = self.lm.unembed_matrix()
        rows = torch.zeros(len(chunk), d)
        for pos, (_, cand) in enumerate(chunk):
            anchor = W[int(cand.ids[0])].float()
            rows[pos] = anchor
            if tg.emit_y.numel() == 0:
                continue
            sel = tg.emit_owner == pos
            n = int(sel.sum())
            if n < 2:
                continue
            H = tg.emit_h[sel]  # [n, d]
            rho = tg.emit_y[sel] - H @ anchor  # [n]
            lam = self.cfg.output_ridge * float(H.pow(2).sum(-1).mean().clamp_min(1e-8)) * max(1.0, d / n)
            A = H.T @ H + lam * torch.eye(d)
            delta = torch.linalg.solve(A, H.T @ rho)
            # Trust region: a correction larger than the anchor itself means the
            # linear model is extrapolating, and an extrapolating output row is
            # exactly the one that fires off-context.
            cap = self.cfg.output_trust_region * float(anchor.norm().clamp_min(1e-6))
            norm = float(delta.norm())
            if norm > cap:
                delta = delta * (cap / norm)
            rows[pos] = anchor + delta
        return rows


def _per_owner_sumsq(r: torch.Tensor, owner: torch.Tensor, C: int) -> torch.Tensor:
    """Sum of squared residuals grouped by candidate.

    Claim: infrastructure.
    """
    sq = (r.float() ** 2).sum(-1)
    out = torch.zeros(C, dtype=torch.float32)
    out.index_add_(0, owner, sq)
    return out


def unigram_frequencies(docs: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Token frequencies for the inverse-frequency composition weights.

    Claim: non-regression — a better initialisation means a smaller residual to
    close, which means less drift to certify.
    """
    from collections import Counter

    c: Counter = Counter()
    for d in docs:
        c.update(d)
    return dict(c)
