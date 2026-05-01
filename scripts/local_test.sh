#!/usr/bin/env bash
#
# local_test.sh - Smoke test the Lambda handler locally before paying for
# an AWS deploy. Uses Docker to mimic the actual Lambda runtime.
#
# This catches stupid issues (missing imports, bad paths, broken JSON parsing)
# without burning ECR push time.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "out/best.pt" ]] || [[ ! -f "data/tokenizer.json" ]]; then
    echo "ERROR: missing model artifacts."
    echo "Run: cd model && python prepare.py && python train.py"
    exit 1
fi

# Stage model artifacts
echo "==> staging model artifacts"
rm -rf model_artifacts
mkdir -p model_artifacts
cp out/best.pt model_artifacts/
cp data/tokenizer.json model_artifacts/

echo "==> building image"
docker buildx build --platform linux/amd64 -t klotho:local -f serving/Dockerfile --load .

echo "==> starting container on :9000"
CONTAINER=$(docker run -d --rm \
    --platform linux/amd64 \
    -p 9000:8080 \
    klotho:local)

cleanup() {
    echo "==> stopping container ${CONTAINER}"
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Lambda's runtime emulator listens on /2015-03-31/functions/function/invocations
# and expects a full Lambda invocation event payload.
sleep 5

echo
echo "==> test 1: GET /health"
curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
    -d '{"requestContext":{"http":{"method":"GET","path":"/health"}}}' \
    | python3 -m json.tool
echo

echo "==> test 2: POST /generate"
curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
    -d '{"requestContext":{"http":{"method":"POST","path":"/generate"}},"body":"{\"prompt\":\"ROMEO:\",\"max_tokens\":50,\"temperature\":0.8,\"top_k\":40}"}' \
    | python3 -m json.tool
echo

echo "==> test 3: GET /history"
curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
    -d '{"requestContext":{"http":{"method":"GET","path":"/history"}}}' \
    | python3 -m json.tool
echo

echo "==> done. local tests passed."
