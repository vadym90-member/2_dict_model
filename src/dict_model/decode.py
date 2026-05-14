"""Decoding utilities: greedy + beam search.

Both routines operate on a *single* ``(id, eng)`` pair at a time. For the
glossary task the input is short and the beam width is small, so the
clearer per-example implementation is plenty fast — vectorising over a
batch would only matter at much larger throughput targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import DictTransformer
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, Tokenizer


@dataclass
class DecodeConfig:
    beam_size: int = 4
    length_penalty: float = 0.6
    max_new_tokens: int = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_source(tokenizer: Tokenizer, raw_id: str, raw_eng: str, device: torch.device):
    ids = tokenizer.encode_source(raw_id, raw_eng)
    src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)        # (1, S)
    src_kpm = src == PAD_ID
    return src, src_kpm


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: DictTransformer,
    tokenizer: Tokenizer,
    raw_id: str,
    raw_eng: str,
    *,
    max_new_tokens: int = 24,
    device: torch.device | None = None,
) -> str:
    """Greedy decoding: at each step pick the argmax token."""
    device = device or next(model.parameters()).device
    model.eval()
    src, src_kpm = _encode_source(tokenizer, raw_id, raw_eng, device)
    memory = model.encode(src, src_kpm)
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        tgt_kpm = ys == PAD_ID
        dec = model.decode(ys, memory, tgt_kpm=tgt_kpm, mem_kpm=src_kpm)
        logits = model.lm_head(dec[:, -1])           # (1, V)
        next_tok = int(logits.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
        if next_tok == EOS_ID:
            break
    return tokenizer.decode_target(ys[0].tolist())


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@dataclass
class _Beam:
    tokens: list[int]
    log_prob: float
    finished: bool = False


@torch.no_grad()
def beam_search_decode(
    model: DictTransformer,
    tokenizer: Tokenizer,
    raw_id: str,
    raw_eng: str,
    *,
    cfg: DecodeConfig = DecodeConfig(),
    device: torch.device | None = None,
) -> str:
    """Beam search with length-penalty rescoring (a la GNMT)."""
    device = device or next(model.parameters()).device
    model.eval()
    src, src_kpm = _encode_source(tokenizer, raw_id, raw_eng, device)
    memory = model.encode(src, src_kpm)
    V = model.cfg.tgt_vocab_size

    beams: list[_Beam] = [_Beam(tokens=[BOS_ID], log_prob=0.0)]
    finished: list[_Beam] = []

    for _ in range(cfg.max_new_tokens):
        live = [b for b in beams if not b.finished]
        if not live:
            break
        # Run all live beams through the decoder as a single batch.
        ys = torch.tensor([b.tokens for b in live], dtype=torch.long, device=device)
        tgt_kpm = ys == PAD_ID
        mem_b = memory.expand(len(live), *memory.shape[1:])
        srcm_b = src_kpm.expand(len(live), *src_kpm.shape[1:])
        dec = model.decode(ys, mem_b, tgt_kpm=tgt_kpm, mem_kpm=srcm_b)
        logits = model.lm_head(dec[:, -1])                   # (Nlive, V)
        log_probs = torch.log_softmax(logits, dim=-1)        # (Nlive, V)
        topk_lp, topk_ix = log_probs.topk(cfg.beam_size, dim=-1)  # each (Nlive, K)

        candidates: list[_Beam] = []
        for i, beam in enumerate(live):
            for k in range(cfg.beam_size):
                tok = int(topk_ix[i, k].item())
                lp = float(topk_lp[i, k].item())
                new_beam = _Beam(
                    tokens=beam.tokens + [tok],
                    log_prob=beam.log_prob + lp,
                    finished=(tok == EOS_ID),
                )
                candidates.append(new_beam)
        # Keep top-K candidates by length-penalised score.
        def score(b: _Beam) -> float:
            L = max(len(b.tokens) - 1, 1)
            lp = ((5.0 + L) / 6.0) ** cfg.length_penalty
            return b.log_prob / lp

        candidates.sort(key=score, reverse=True)
        beams = candidates[: cfg.beam_size]
        for b in list(beams):
            if b.finished:
                finished.append(b)
        beams = [b for b in beams if not b.finished]
        if len(finished) >= cfg.beam_size:
            break

    pool = finished if finished else beams

    def score(b: _Beam) -> float:
        L = max(len(b.tokens) - 1, 1)
        lp = ((5.0 + L) / 6.0) ** cfg.length_penalty
        return b.log_prob / lp

    best = max(pool, key=score)
    return tokenizer.decode_target(best.tokens)
