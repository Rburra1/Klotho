"""
Download Tiny Shakespeare and prepare train/val token splits.

Output:
  data/input.txt        raw text
  data/train.bin        uint16 array of token ids, training split
  data/val.bin          uint16 array of token ids, validation split
  data/tokenizer.json   character vocabulary
  data/meta.json        metadata: vocab_size, num train tokens, num val tokens
"""

import json
import os
from pathlib import Path
import urllib.request

import numpy as np

from tokenizer import CharTokenizer


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already have {dest}")
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "input.txt"
    download(SHAKESPEARE_URL, raw_path)
    text = raw_path.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    # Build tokenizer over the entire corpus.
    tok = CharTokenizer.from_text(text)
    print(f"vocab size: {tok.vocab_size}")
    tok.save(DATA_DIR / "tokenizer.json")

    # Encode and split 90/10.
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n_train = int(0.9 * len(ids))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens:   {len(val_ids):,}")

    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")

    meta = {
        "vocab_size": tok.vocab_size,
        "num_train_tokens": int(len(train_ids)),
        "num_val_tokens": int(len(val_ids)),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {DATA_DIR}/{{train.bin,val.bin,tokenizer.json,meta.json}}")


if __name__ == "__main__":
    main()
