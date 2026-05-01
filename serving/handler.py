"""
Athena - serverless Klotho inference API on AWS Lambda.

Endpoints:
  POST /generate   { "prompt": str, "max_tokens": int, "temperature": float, "top_k": int }
                   -> { "id": str, "prompt": str, "completion": str, "latency_ms": int, ... }
  GET  /history    [ ?limit=N ]
                   -> { "items": [ ... ] }
  GET  /health     -> { "ok": true, "model_loaded": bool }

The handler is loaded once per Lambda container (cold start) and reused across
warm invocations. Model weights live alongside the code in the container
image. SQLite database lives at /tmp/athena.db (Lambda's writable scratch).

History is best-effort: each Lambda container has its own /tmp, so the SQLite
file is local to a single warm container. For a real production deployment
this would be backed by RDS or DynamoDB. Documented in the README.
"""

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Make local imports work when packaged in the Lambda container.
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

from model import Klotho, KlothoConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


# ---------- logging ----------

logger = logging.getLogger("athena")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    # CloudWatch ingests stdout line-by-line and parses JSON if structured.
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)


def log(event: str, **fields: Any) -> None:
    """Structured JSON log line for CloudWatch."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload))


# ---------- model loading (cold start) ----------

MODEL_DIR = Path(os.environ.get("ATHENA_MODEL_DIR", "/var/task/model_artifacts"))
DB_PATH = os.environ.get("ATHENA_DB_PATH", "/tmp/athena.db")

_model: Klotho | None = None
_tokenizer: CharTokenizer | None = None
_device: torch.device | None = None
_model_meta: dict[str, Any] = {}


def _load_model() -> None:
    """Load the trained Klotho checkpoint. Called once per container."""
    global _model, _tokenizer, _device, _model_meta

    start = time.time()
    _device = torch.device("cpu")  # Lambda has no GPU; CPU inference only

    tokenizer_path = MODEL_DIR / "tokenizer.json"
    ckpt_path = MODEL_DIR / "best.pt"

    if not tokenizer_path.exists() or not ckpt_path.exists():
        log(
            "model_load_error",
            tokenizer_exists=tokenizer_path.exists(),
            checkpoint_exists=ckpt_path.exists(),
            model_dir=str(MODEL_DIR),
        )
        raise RuntimeError(
            f"Missing model artifacts in {MODEL_DIR}. Need tokenizer.json and best.pt."
        )

    _tokenizer = CharTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    cfg = KlothoConfig(**ckpt["config"])
    _model = Klotho(cfg).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    _model_meta = {
        "params_m": _model.num_params() / 1e6,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "block_size": cfg.block_size,
        "vocab_size": cfg.vocab_size,
        "val_loss": ckpt.get("val_loss"),
        "iter": ckpt.get("iter"),
    }

    log("model_loaded", load_ms=int((time.time() - start) * 1000), **_model_meta)


# Load eagerly at module import time (counts toward cold start).
try:
    _load_model()
except Exception as exc:
    log("model_load_failed", error=str(exc))


# ---------- sqlite ----------

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    """Create tables if they don't exist. Idempotent."""
    with db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS inferences (
                id           TEXT PRIMARY KEY,
                created_at   INTEGER NOT NULL,
                prompt       TEXT NOT NULL,
                completion   TEXT NOT NULL,
                temperature  REAL NOT NULL,
                top_k        INTEGER NOT NULL,
                max_tokens   INTEGER NOT NULL,
                tokens_out   INTEGER NOT NULL,
                latency_ms   INTEGER NOT NULL,
                request_id   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_inferences_created
                ON inferences (created_at DESC);

            CREATE TABLE IF NOT EXISTS daily_stats (
                day          TEXT PRIMARY KEY,
                request_count INTEGER NOT NULL DEFAULT 0,
                tokens_total  INTEGER NOT NULL DEFAULT 0,
                p50_latency   INTEGER,
                p95_latency   INTEGER
            );
        """)


init_schema()


def record_inference(
    *,
    id: str,
    prompt: str,
    completion: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    tokens_out: int,
    latency_ms: int,
    request_id: str | None,
) -> None:
    now = int(time.time())
    today = time.strftime("%Y-%m-%d", time.gmtime(now))

    with db() as c:
        c.execute(
            """
            INSERT INTO inferences
                (id, created_at, prompt, completion, temperature, top_k,
                 max_tokens, tokens_out, latency_ms, request_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (id, now, prompt, completion, temperature, top_k,
             max_tokens, tokens_out, latency_ms, request_id),
        )

        # Update daily aggregate. Real production would use a batch job.
        c.execute(
            """
            INSERT INTO daily_stats (day, request_count, tokens_total)
            VALUES (?, 1, ?)
            ON CONFLICT(day) DO UPDATE SET
                request_count = request_count + 1,
                tokens_total  = tokens_total + excluded.tokens_total
            """,
            (today, tokens_out),
        )


def fetch_history(limit: int = 20) -> list[dict[str, Any]]:
    with db() as c:
        rows = c.execute(
            """
            SELECT id, created_at, prompt, completion, temperature, top_k,
                   max_tokens, tokens_out, latency_ms
              FROM inferences
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- inference ----------

def generate(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> tuple[str, int, int]:
    """Returns (completion, tokens_out, latency_ms)."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    start = time.time()

    encoded = _tokenizer.encode(prompt) or _tokenizer.encode("\n")
    idx = torch.tensor([encoded], dtype=torch.long, device=_device)

    with torch.no_grad():
        out = _model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    new_ids = out[0].tolist()[len(encoded):]
    completion = _tokenizer.decode(new_ids)
    latency_ms = int((time.time() - start) * 1000)
    return completion, len(new_ids), latency_ms


# ---------- request helpers ----------

def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def _bad_request(msg: str) -> dict[str, Any]:
    return _response(400, {"error": msg})


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


# ---------- handlers ----------

def handle_health(_event: dict[str, Any]) -> dict[str, Any]:
    return _response(200, {
        "ok": True,
        "model_loaded": _model is not None,
        "model": _model_meta,
    })


def handle_generate(event: dict[str, Any]) -> dict[str, Any]:
    request_id = (
        event.get("requestContext", {}).get("requestId")
        or event.get("headers", {}).get("x-amzn-requestid")
    )

    try:
        body = _parse_body(event)
    except json.JSONDecodeError:
        return _bad_request("Invalid JSON body")

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return _bad_request("prompt must be a non-empty string")

    max_tokens = int(body.get("max_tokens", 200))
    temperature = float(body.get("temperature", 0.8))
    top_k = int(body.get("top_k", 40))

    if max_tokens < 1 or max_tokens > 1000:
        return _bad_request("max_tokens must be between 1 and 1000")
    if temperature <= 0 or temperature > 2.0:
        return _bad_request("temperature must be in (0, 2]")
    if top_k < 1 or top_k > 256:
        return _bad_request("top_k must be in [1, 256]")

    inference_id = str(uuid.uuid4())

    log(
        "generate_start",
        id=inference_id,
        request_id=request_id,
        prompt_len=len(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    try:
        completion, tokens_out, latency_ms = generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k
        )
    except Exception as exc:
        log("generate_error", id=inference_id, error=str(exc))
        return _response(500, {"error": "inference failed"})

    try:
        record_inference(
            id=inference_id,
            prompt=prompt,
            completion=completion,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            request_id=request_id,
        )
    except Exception as exc:
        # Persistence failure shouldn't fail the request.
        log("history_write_failed", id=inference_id, error=str(exc))

    log(
        "generate_done",
        id=inference_id,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )

    return _response(200, {
        "id": inference_id,
        "prompt": prompt,
        "completion": completion,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "model": _model_meta.get("params_m"),
    })


def handle_history(event: dict[str, Any]) -> dict[str, Any]:
    qs = event.get("queryStringParameters") or {}
    try:
        limit = int(qs.get("limit", 20))
    except ValueError:
        return _bad_request("limit must be an integer")
    limit = max(1, min(limit, 100))

    items = fetch_history(limit=limit)
    return _response(200, {"items": items, "count": len(items)})


# ---------- entry point ----------

ROUTES = {
    ("GET", "/health"): handle_health,
    ("POST", "/generate"): handle_generate,
    ("GET", "/history"): handle_history,
}


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """API Gateway HTTP API v2 entry point."""
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod")
              or "GET")
    path = (event.get("requestContext", {}).get("http", {}).get("path")
            or event.get("rawPath")
            or event.get("path")
            or "/")

    handler = ROUTES.get((method, path))
    if handler is None:
        return _response(404, {"error": f"no route for {method} {path}"})

    return handler(event)
