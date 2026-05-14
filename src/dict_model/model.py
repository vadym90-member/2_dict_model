"""Transformer encoder-decoder, built from PyTorch primitives.

The architecture is the standard Vaswani-style seq2seq Transformer:
sinusoidal positional encodings, multi-head scaled dot-product attention,
position-wise feed-forward blocks, residual connections with pre-norm. No
weights are imported from any pretrained model — every parameter is
initialised randomly (Xavier-uniform for projection matrices).

Tensor conventions
------------------
We use *batch-first* throughout (``B, L, D``). Attention masks follow PyTorch
nn.functional.scaled_dot_product_attention semantics:

  - ``key_padding_mask``: bool of shape (B, L_key), True means *padding*
    (i.e. positions to ignore).
  - causal mask is built on-the-fly inside the decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sin/cos positional encoding, registered as a non-trainable buffer."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with separate Q/K/V projections.

    The forward pass accepts:
      * ``key_padding_mask`` (B, Lk) — True at positions to ignore
      * ``attn_mask``        (Lq, Lk) — additive mask (used for causal mask)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(proj.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, Dh)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(q))
        K = self._split_heads(self.W_k(k))
        V = self._split_heads(self.W_v(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,Lq,Lk)
        if attn_mask is not None:
            scores = scores + attn_mask  # (Lq,Lk) broadcasts
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk) True=pad. Expand to (B,1,1,Lk).
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B,H,Lq,Dh)
        return self.W_o(self._merge_heads(out))


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Encoder / decoder layers
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kpm: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=kpm))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_kpm: torch.Tensor | None,
        mem_kpm: torch.Tensor | None,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, key_padding_mask=tgt_kpm, attn_mask=causal_mask))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, key_padding_mask=mem_kpm))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Full seq2seq model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 64
    share_target_embedding: bool = True


class DictTransformer(nn.Module):
    """Encoder-decoder Transformer mapping (id, eng) tokens to Korean tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=0)
        nn.init.normal_(self.src_embed.weight, mean=0.0, std=cfg.d_model ** -0.5)
        nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.src_embed.weight[0].zero_()
            self.tgt_embed.weight[0].zero_()

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_dec_layers)]
        )
        self.norm_enc = nn.LayerNorm(cfg.d_model)
        self.norm_dec = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.share_target_embedding:
            self.lm_head.weight = self.tgt_embed.weight
        else:
            nn.init.xavier_uniform_(self.lm_head.weight)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """Additive causal mask: 0 where allowed, -inf where forbidden."""
        m = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.cfg.d_model)
        return self.dropout(self.pos_enc(x))

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.cfg.d_model)
        return self.dropout(self.pos_enc(x))

    # -- forward -----------------------------------------------------------

    def encode(self, src: torch.Tensor, src_kpm: torch.Tensor) -> torch.Tensor:
        h = self._embed_src(src)
        for layer in self.encoder:
            h = layer(h, src_kpm)
        return self.norm_enc(h)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_kpm: torch.Tensor,
        mem_kpm: torch.Tensor,
    ) -> torch.Tensor:
        L = tgt_in.size(1)
        cmask = self.causal_mask(L, tgt_in.device)
        h = self._embed_tgt(tgt_in)
        for layer in self.decoder:
            h = layer(h, memory, tgt_kpm=tgt_kpm, mem_kpm=mem_kpm, causal_mask=cmask)
        return self.norm_dec(h)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        *,
        src_kpm: torch.Tensor,
        tgt_kpm: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_kpm)
        dec = self.decode(tgt_in, memory, tgt_kpm=tgt_kpm, mem_kpm=src_kpm)
        return self.lm_head(dec)  # (B, T-1, V)

    # -- parameter count for logging --------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
