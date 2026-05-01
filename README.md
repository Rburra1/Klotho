# Klotho

A tiny GPT-style transformer, written from scratch in PyTorch and deployed as a serverless inference API on AWS.

> Named after Klotho (Κλωθώ), the Greek Fate who spins the thread of life — one fiber at a time, the way an autoregressive language model spins text one token at a time.

This repo is the full stack:

- **`model/`** — the architecture, trained from scratch on Tiny Shakespeare in ~96 min on Apple Silicon
- **`serving/`** — Lambda handler that loads the trained checkpoint and exposes it over HTTP
- **`infra/`** — Terraform for the AWS deployment (Lambda, API Gateway, IAM, CloudWatch, budget alert)
- **`scripts/`** — one-command deploy, local Docker test, full teardown

Companion to [Strand](https://strand.rithvikburra.com), the chat surface this kind of model serves.

## What's in the model

10M-param decoder-only transformer, GPT-2 style. ~200 lines of model code in `model/model.py`. No Hugging Face, no third-party transformer abstractions — just the math.

- 6 layers, 6 attention heads per layer, 384 embedding dim
- Pre-LayerNorm transformer blocks
- Multi-head causal self-attention with single fused QKV projection
- 4× expansion FFN with GELU
- Token + learned positional embeddings, summed at input
- Weight tying between input embedding and output projection
- GPT-2 init: N(0, 0.02), residual projections scaled by 1/√(2L)

Trained 5,000 iterations on Tiny Shakespeare via Apple Silicon MPS. Best validation loss **1.49**. Output preserves speaker formatting, iambic cadence, archaic vocabulary, and cross-corpus character recall. Coherence is local (~5 words) — expected ceiling for this scale.

## What's in the deployment

- **AWS Lambda** running a containerized PyTorch + Klotho checkpoint (~700 MB image)
- **API Gateway HTTP API** with three routes (`/health`, `/generate`, `/history`)
- **CloudWatch** with structured JSON logging and 7-day retention
- **SQLite** inside the Lambda for inference history and per-day stats
- **ECR** for hosting the container image, with a 5-image lifecycle policy
- **IAM** scoped to `AWSLambdaBasicExecutionRole` only
- **Terraform** for everything — no clicking around the AWS Console
- **$1/month budget alert** as cost insurance

Designed to stay in the AWS Free Tier indefinitely at portfolio scale. Expected monthly cost: $0.

## API

### `POST /generate`

```bash
curl -X POST $API_URL/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ROMEO:","max_tokens":200,"temperature":0.8,"top_k":40}'
```

Response:

```json
{
  "id": "uuid-v4",
  "prompt": "ROMEO:",
  "completion": "I would thy love leave to come to see thee...",
  "tokens_out": 200,
  "latency_ms": 2340,
  "model": 10.73
}
```

### `GET /history?limit=20`

Returns recent inferences. Note: history is per-Lambda-container (`/tmp/athena.db`), not global. For real production, this would be backed by RDS or DynamoDB. Documented as a known limitation.

### `GET /health`

Returns model metadata and load status.

## Quick start

### Train the model

```bash
cd model
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt

python prepare.py    # downloads Tiny Shakespeare, writes data/
python train.py      # ~96 min on Apple M4, writes out/best.pt
python sample.py "ROMEO:" --stream
```

### Test the deployment locally

```bash
./scripts/local_test.sh
```

Spins up the Lambda runtime emulator in Docker on `localhost:9000` and exercises all three endpoints. Catches deployment issues before you push to ECR.

### Deploy to AWS

Read [docs/DEPLOY.md](docs/DEPLOY.md) end to end. Quick version:

```bash
./scripts/deploy.sh
```

This builds the container image, pushes to ECR, runs Terraform, and rolls the Lambda function. Idempotent — run any time.

### Tear down

```bash
./scripts/destroy.sh
```

Removes all AWS resources. Returns billing to $0.

## File map

```
model/
  model.py         Klotho architecture (~200 lines)
  tokenizer.py     Char-level tokenizer with save/load
  prepare.py       Download Tiny Shakespeare, encode, write train/val .bin
  train.py         Training loop with MPS/CUDA/CPU device selection
  sample.py        Load checkpoint and generate text
  requirements.txt Training-time dependencies
serving/
  handler.py       Lambda handler: routes, generation, SQLite, structured logging
  Dockerfile       Lambda container image (Python 3.11 + PyTorch CPU)
infra/
  main.tf          Full Terraform: ECR, IAM, Lambda, API Gateway, CloudWatch, budget
  terraform.tfvars Region, project name, alert email
scripts/
  deploy.sh        Build + push + Terraform apply + roll Lambda
  local_test.sh    Run the handler locally via Lambda runtime emulator
  destroy.sh       Tear down everything with one command
docs/
  DEPLOY.md        Step-by-step AWS deploy walkthrough
data/              [gitignored] Tokenized corpus (created by prepare.py)
out/               [gitignored] Training checkpoints (created by train.py)
SAMPLES.md         Generated samples from the trained model
```

## Engineering notes

- **Cold start ≈ 8–15s.** PyTorch + 10MB checkpoint + container init. Acceptable for a portfolio API; for a real product use provisioned concurrency or pre-warm.
- **Reserved concurrency = 5.** Caps simultaneous Lambda executions, which caps cost in pathological cases.
- **API Gateway throttling** at 5 req/s sustained, 10 burst.
- **`x86_64` Lambda, not Graviton.** PyTorch wheels ship for x86_64. Apple Silicon Macs cross-compile via `--platform linux/amd64` in the deploy script.
- **No NAT Gateway, no VPC.** NAT is the most common surprise AWS bill. Lambda runs outside any VPC for this project.
- **SQLite over RDS.** Free, real SQL, but per-container persistence. Real production would use RDS or DynamoDB.

## Roadmap

- BPE tokenization for richer vocab
- Train on a larger corpus (50–100MB, mixed sources)
- Swap SQLite for RDS Postgres with Alembic migrations
- ONNX export for browser inference (HIPAA-relevant: patient data never leaves client)
- Streaming responses via API Gateway WebSocket
- Custom domain with ACM certificate

## Cost

| Resource | Cost | Free tier coverage |
|----------|------|---------------------|
| Lambda invocations | $0.20 / 1M requests | 1M/month free **forever** |
| Lambda compute | $0.0000166667 / GB-second | 400K GB-seconds/month free **forever** |
| API Gateway HTTP API | $1.00 / 1M requests | 1M/month free for 12 months |
| CloudWatch Logs ingestion | $0.50 / GB | 5 GB/month free |
| ECR storage | $0.10 / GB-month | 500 MB free for 12 months |
| Budget alerts | Free | Unlimited |

For a portfolio API receiving ~50 requests over a 4-week window: **$0.00**.
