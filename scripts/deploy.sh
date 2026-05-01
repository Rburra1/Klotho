#!/usr/bin/env bash
#
# deploy.sh - Build the Lambda container image, push to ECR, and roll the
# Lambda function to use the new image.
#
# Prereqs (one-time):
#   - aws cli v2 installed and configured (aws configure)
#   - docker installed and running
#   - terraform installed
#   - You have already trained Klotho: out/best.pt and data/tokenizer.json exist
#
# Usage:
#   ./scripts/deploy.sh
#
# Idempotent. Run any time you change source files or want to redeploy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- config ----------

REGION="${AWS_REGION:-us-east-1}"
PROJECT="klotho"

# Resolve account ID from current AWS credentials.
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_REPO="${ECR_HOST}/${PROJECT}"

echo "==> account=${ACCOUNT_ID} region=${REGION} repo=${ECR_REPO}"

# ---------- 1. stage model artifacts into the build context ----------

echo "==> staging model artifacts"
rm -rf model_artifacts
mkdir -p model_artifacts

if [[ ! -f "out/best.pt" ]]; then
    echo "ERROR: out/best.pt not found."
    echo "Train Klotho first: cd model && python prepare.py && python train.py"
    exit 1
fi
if [[ ! -f "data/tokenizer.json" ]]; then
    echo "ERROR: data/tokenizer.json not found."
    echo "Run: cd model && python prepare.py"
    exit 1
fi

cp out/best.pt model_artifacts/
cp data/tokenizer.json model_artifacts/
echo "    artifacts: $(du -sh model_artifacts | cut -f1)"

# ---------- 2. ensure ECR repo exists (terraform apply will also create it,
# but on first run we need it before docker push). ----------

if ! aws ecr describe-repositories --repository-names "${PROJECT}" --region "${REGION}" >/dev/null 2>&1; then
    echo "==> creating ECR repository (first run)"
    cd infra
    terraform init -upgrade
    terraform apply -target=aws_ecr_repository.klotho -auto-approve
    cd ..
fi

# ---------- 3. login to ECR ----------

echo "==> logging in to ECR"
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ECR_HOST}"

# ---------- 4. build container image (linux/amd64 to match Lambda) ----------

# Lambda runs on x86_64 by default. If you're on an Apple Silicon Mac, we need
# --platform linux/amd64 to cross-build. The container will be slower to build
# under emulation but the resulting image runs natively on Lambda.
echo "==> building image"
docker buildx build \
    --platform linux/amd64 \
    -t "${PROJECT}:latest" \
    -f serving/Dockerfile \
    --load \
    .

# ---------- 5. tag and push ----------

echo "==> tagging and pushing"
docker tag "${PROJECT}:latest" "${ECR_REPO}:latest"
docker push "${ECR_REPO}:latest"

# Capture the pushed digest so the Lambda update is deterministic
DIGEST=$(aws ecr describe-images \
    --repository-name "${PROJECT}" \
    --region "${REGION}" \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imageDigest' \
    --output text)
echo "    image digest: ${DIGEST}"

# ---------- 6. terraform apply for everything else ----------

echo "==> terraform apply"
cd infra
terraform init -upgrade
terraform apply -auto-approve
cd ..

# ---------- 7. force the Lambda to pick up the new image
# (terraform doesn't detect tag changes when image_tag stays at 'latest') ----------

echo "==> updating Lambda function code to latest image"
aws lambda update-function-code \
    --function-name "${PROJECT}" \
    --image-uri "${ECR_REPO}:latest" \
    --region "${REGION}" >/dev/null

echo "==> waiting for Lambda update to complete"
aws lambda wait function-updated \
    --function-name "${PROJECT}" \
    --region "${REGION}"

# ---------- 8. emit the public URL ----------

API_URL=$(cd infra && terraform output -raw api_endpoint)
echo
echo "==> deployed."
echo "    api: ${API_URL}"
echo
echo "    test:"
echo "      curl ${API_URL}/health"
echo "      curl -X POST ${API_URL}/generate \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"prompt\":\"ROMEO:\",\"max_tokens\":200}'"
