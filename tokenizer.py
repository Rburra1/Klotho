"""
Character-level tokenizer for Klotho.

Each unique character in the training text becomes a token. This avoids the
complexity of BPE for v1, at the cost of longer sequences. Output text from a
char-level model is recognizably language-like at the character level even
when not coherent at the sentence level - exactly the right benchmark for a
tiny model trained on a small corpus.

To swap in a real BPE tokenizer later, replace this module's encode/decode
functions and the vocab_size used in KlothoConfig. The model code itself doesn't
care about tokenization scheme.
"""

import json
import os
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(set(text)))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])

    def save(self, path: str | Path) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)
