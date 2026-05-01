# Klotho

A tiny GPT-style transformer, written from scratch in PyTorch, trained on a single MacBook in a few hours.

Companion to [Strand](https://strand.rithvikburra.com). Where Strand is the cloud-API chat surface, Klotho is the from-scratch model: ~10 million parameters, character-level, no Hugging Face, no third-party model code. Just the math.

> Named after Klotho (Κλωθώ), the Greek Fate who spins the thread of life - one fiber at a time, the way an autoregressive language model spins text one token at a time.

## Why

To prove out the basics of training a real language model end to end without leaning on `transformers`, `accelerate`, or any other framework that hides what's actually happening. The model code in `model.py` is the actual code that runs - no inheritance hierarchies, no abstractions to follow into. About 200 lines.

The 10M-param scale is intentional. It's small enough to train on an Apple Silicon laptop in a few hours via the MPS backend, and it's the smallest size where the architectural choices (multi-head attention, pre-norm residuals, weight tying, GELU, AdamW + cosine LR) start to matter for output quality. Output is recognizable Shakespeare-style prose at the character level - rhythmic, archaic, syntactically coherent over short spans, but obviously not coherent at the sentence level. That's the right benchmark for this size.

## Architecture

Decoder-only transformer, GPT-2 style:

- 6 layers, 6 attention heads per layer, 384 embedding dim
- Pre-LayerNorm: norm before each sublayer, residual added after
- Multi-head causal self-attention, causal mask applied to attention scores before softmax
- 4× expansion feed-forward with GELU
- Token + learned positional embeddings, summed at input
- Weight tying between input embedding and output projection
- Standard GPT-2 init: N(0, 0.02), residual projections scaled by 1/√(2L)

Total params: ~10.7M.

## Training

```bash
pip install -r requirements.txt
python prepare.py        # downloads Tiny Shakespeare, builds char vocab, writes train/val binary splits
python train.py          # ~2-3 hours on M4 with MPS, saves best.pt to out/
python sample.py "ROMEO:" --stream
```

Training loop:
- Batch size 64, context length 256
- AdamW, lr 3e-4 → 3e-5 with linear warmup (200 steps) and cosine decay
- Gradient clipping at 1.0, weight decay 0.1, no decay on biases or LayerNorm
- 5000 iterations, eval every 250 steps on 50 batches
- Checkpoints best validation loss to `out/best.pt`

Expected validation loss on Tiny Shakespeare: ~1.45 nats/token. Below 1.5 is where the output starts looking like real Shakespeare excerpts.

## Sampling

```bash
python sample.py "QUEEN:"
python sample.py "JULIET:" --max-new-tokens 1000 --temperature 0.7 --top-k 40 --stream
```

`--stream` prints character by character as the model generates, which mirrors the streaming UI in Strand. `--temperature` controls randomness (0.8 is a good default; higher = wilder, lower = more deterministic). `--top-k` restricts each sampling step to the top K most-likely tokens.

## File map

```
model.py        Klotho + KlothoConfig: the transformer itself, ~200 lines
tokenizer.py    Character-level tokenizer with save/load
prepare.py      Download Tiny Shakespeare, encode, write train/val .bin
train.py        Training loop with MPS/CUDA/CPU device selection
sample.py       Load checkpoint and generate text
data/           Created by prepare.py: input.txt, train.bin, val.bin, tokenizer.json, meta.json
out/            Created by train.py: best.pt + periodic iter_*.pt checkpoints
```

## Swapping in your own corpus

Replace `data/input.txt` with any text file (single corpus, ASCII or UTF-8), then re-run `prepare.py` and `train.py`. For corpora under ~1MB the model will overfit; aim for at least 5MB. For high-quality output on real text, switch to a BPE tokenizer (drop in `tiktoken` or train a `sentencepiece` model) and bump `vocab_size` accordingly. The model code is tokenizer-agnostic.

## What this is not

- A useful chatbot. 10M-param char-level models trained on 1MB of text don't carry meaning; they carry style. Feed it "ROMEO:" and you get plausible-looking iambic pentameter that doesn't mean anything.
- A reference implementation. For that, see Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) - this borrows the same architectural decisions but is rewritten cleanly for clarity.
- A finetuned model. There is no instruction-tuning, no RLHF, no preference data. It's pure pretraining on next-character prediction.

## Roadmap

- BPE tokenization for richer vocab
- Train on a larger corpus (20-100MB, mixed sources)
- Export to ONNX for browser inference via `onnxruntime-web`
- Wire into Strand as a "local mode" toggle: same UI, on-device inference, no API calls

The browser-inference step is the interesting one - it would let Strand demo a path where patient or proprietary data never leaves the client. For HIPAA-bound contexts, that's the actual value proposition.
