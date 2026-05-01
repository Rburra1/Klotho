#!/usr/bin/env bash
#
# destroy.sh - Tear down all AWS infrastructure for Athena.
#
# Run this AFTER you've gotten a decision back from Pear (or any time you
# want to stop the project entirely). Removes:
#   - Lambda function
#   - API Gateway
#   - ECR repository (with all images)
#   - IAM role
#   - CloudWatch log group
#   - Budget alert
#
# After this completes, AWS billing should show $0.00 going forward.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/infra"

echo "==> About to destroy all Athena AWS resources."
echo
read -p "Type 'destroy' to confirm: " confirm
if [[ "$confirm" != "destroy" ]]; then
    echo "Aborted."
    exit 1
fi

terraform destroy -auto-approve

echo
echo "==> done. Verify in AWS Console: https://console.aws.amazon.com/billing/home"
