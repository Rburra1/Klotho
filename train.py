"""
Train Klotho on tokenized data.

Defaults are tuned for an Apple Silicon MacBook (M1/M2/M3/M4) using the MPS
backend. On a 16GB M4 the small config below trains in ~2-3 hours and reaches
val loss around 1.45 on Tiny Shakespeare, which produces recognizable
Shakespeare-style character output.

Usage:
  python prepare.py
  python train.py
  python sample.py "ROMEO:"
"""

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import Klotho, KlothoConfig
from tokenizer import CharTokenizer


# ---------- config ----------

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "out"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1

LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 200
LR_DECAY_ITERS = 5000
MAX_ITERS = 5000

GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)

EVAL_INTERVAL = 250
EVAL_ITERS = 50
LOG_INTERVAL = 25
SAVE_INTERVAL = 1000


def get_device() -> torch.device:
    """Prefer Apple MPS, fall back to CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(it: int) -> float:
    """Linear warmup then cosine decay."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it >= LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def get_batch(split: str, train_data: np.ndarray, val_data: np.ndarray, device: torch.device):
    """Sample a random batch of (input, target) pairs from the binary data."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    """Run EVAL_ITERS forward passes on each split, return mean loss."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    device = get_device()
    print(f"device: {device}")

    # Load tokenizer + data.
    tok = CharTokenizer.load(DATA_DIR / "tokenizer.json")
    train_data = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
    val_data = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    # Build model.
    cfg = KlothoConfig(
        block_size=BLOCK_SIZE,
        vocab_size=tok.vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=False,
    )
    model = Klotho(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M")

    # AdamW with separate weight-decay groups.
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Don't decay biases or LayerNorm weights.
        if p.dim() < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=BETAS,
    )

    # Training loop.
    start = time.time()
    best_val = float("inf")
    for it in range(MAX_ITERS + 1):
        # Update LR.
        lr = get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Periodic eval.
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - start
            print(
                f"iter {it:5d} | lr {lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"{elapsed:.0f}s elapsed"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "iter": it,
                    "val_loss": best_val,
                }
                torch.save(ckpt, CKPT_DIR / "best.pt")

        # One training step.
        X, Y = get_batch("train", train_data, val_data, device)
        _, loss = model(X, Y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if it % LOG_INTERVAL == 0 and it > 0:
            print(f"  step {it:5d} | loss {loss.item():.4f}")

        if it > 0 and it % SAVE_INTERVAL == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "iter": it,
                },
                CKPT_DIR / f"iter_{it}.pt",
            )

    print(f"done. best val loss: {best_val:.4f}")
    print(f"checkpoint: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
