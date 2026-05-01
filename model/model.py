"""
Klotho - tiny GPT-style transformer, written from scratch.

Architecture:
  - Decoder-only transformer (causal language model)
  - Pre-LayerNorm blocks: LN -> attention -> residual, LN -> FFN -> residual
  - Multi-head causal self-attention with single QKV linear
  - 4x expansion feed-forward with GELU
  - Token + learned positional embeddings, summed at input
  - Weight tying: input embedding shares weights with output projection

This is a minimal but real implementation - no shortcuts in the math.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KlothoConfig:
    block_size: int = 256        # max context length in tokens
    vocab_size: int = 65         # set to actual tokenizer vocab at runtime
    n_layer: int = 6             # number of transformer blocks
    n_head: int = 6              # number of attention heads
    n_embd: int = 384            # embedding / hidden dimension
    dropout: float = 0.1         # dropout probability
    bias: bool = False           # whether linear layers use bias terms


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    All heads computed in parallel via a single (3 * n_embd) linear and
    a head-axis reshape. The causal mask is enforced by upper-triangular
    masking of the attention scores before the softmax.
    """

    def __init__(self, cfg: KlothoConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Combined Q, K, V projection.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection back to model dim.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: lower-triangular ones, shape (1, 1, block, block).
        # Registered as a buffer so it moves with .to(device) but isn't a parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels (n_embd)

        # Project to Q, K, V then split heads.
        # qkv shape: (B, T, 3 * n_embd)  ->  three (B, T, n_embd) tensors
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Reshape each to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask: positions ahead of current token get -inf.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum: (B, n_head, T, head_dim)
        y = att @ v

        # Re-merge heads: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection + dropout.
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward with 4x expansion and GELU activation."""

    def __init__(self, cfg: KlothoConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: attention + FFN with pre-norm and residuals."""

    def __init__(self, cfg: KlothoConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: norm before each sublayer, residual added after.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Klotho(nn.Module):
    """Tiny GPT-style language model."""

    def __init__(self, cfg: KlothoConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_final = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share embedding matrix with output projection.
        # This roughly halves the parameter count of the small models we train.
        self.head.weight = self.token_emb.weight

        # Init weights with the GPT-2 scheme.
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) for stability.
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias) if module.bias is not None else None
            nn.init.ones_(module.weight)

    def num_params(self) -> int:
        """Total trainable params, excluding tied output head."""
        n = sum(p.numel() for p in self.parameters())
        # Subtract the duplicated head weight that's tied to token_emb.
        n -= self.head.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) long tensor of token ids. Returns (logits, loss)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} > block_size {self.cfg.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        tok_e = self.token_emb(idx)   # (B, T, n_embd)
        pos_e = self.pos_emb(pos)     # (1, T, n_embd)
        x = self.drop(tok_e + pos_e)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)         # (B, T, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation. idx: (B, T) seed token ids. Returns extended sequence."""
        for _ in range(max_new_tokens):
            # Crop context if it grows past block_size.
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # Take logits at the final timestep, scale by temperature.
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
