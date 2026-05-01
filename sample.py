"""
Generate text from a trained Klotho checkpoint.

Usage:
  python sample.py "ROMEO:"
  python sample.py "JULIET:" --max-new-tokens 1000 --temperature 0.8 --top-k 40
"""

import argparse
import sys
from pathlib import Path

import torch

from model import Klotho, KlothoConfig
from tokenizer import CharTokenizer


DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "out"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="\n", help="seed text")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "best.pt"))
    p.add_argument("--stream", action="store_true", help="print tokens as they generate")
    args = p.parse_args()

    device = get_device()
    tok = CharTokenizer.load(DATA_DIR / "tokenizer.json")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = KlothoConfig(**ckpt["config"])
    model = Klotho(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encoded = tok.encode(args.prompt)
    if not encoded:
        # Fall back to newline if prompt has no in-vocab characters.
        encoded = tok.encode("\n")
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)

    if args.stream:
        # Token-by-token streaming generation.
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
                logits, _ = model(cond)
                logits = logits[:, -1, :] / max(args.temperature, 1e-8)
                if args.top_k:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
                ch = tok.decode([next_id.item()])
                print(ch, end="", flush=True)
        print()
    else:
        # Batch generation, single decode at the end.
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        text = tok.decode(out[0].tolist()[len(encoded):])
        print(text)


if __name__ == "__main__":
    main()
