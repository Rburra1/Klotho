# Step-by-step deploy

You haven't used AWS before. This walks through the whole thing.

Total time: ~45 min the first time, ~5 min after that.

## 1. Make sure you have a trained Klotho checkpoint

Before deploying, you need `out/best.pt` and `data/tokenizer.json` at the repo root. If you've already trained Klotho overnight, skip to step 2. Otherwise:

```bash
cd model
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt

python prepare.py    # 5 sec, downloads Shakespeare
python train.py      # 2-3 hours on Apple M4
```

When done, you should see `out/best.pt` and `data/tokenizer.json` at the repo root.

## 2. Create AWS account (10 min)

1. Go to https://aws.amazon.com/free and click "Create a Free Account"
2. Email, password, AWS account name (e.g. `rithvik-personal`)
3. **Account type: Personal** (not Business)
4. Add billing info — credit card required even for free tier (this is normal; the budget alert in our Terraform config protects you)
5. Phone verification
6. Choose **Basic Support — Free** plan
7. Sign in to the AWS Console

## 3. Set up the AWS CLI (5 min)

Mac:

```bash
brew install awscli
```

Then create an access key:

1. AWS Console → top right (your name) → **Security credentials**
2. Scroll down to **Access keys** → **Create access key**
3. Choose **Command Line Interface (CLI)**
4. Acknowledge the warning, click Next
5. Description tag: `local-dev`
6. **Save the Access Key ID and Secret Access Key somewhere secure** — you can't view the secret again

Then in terminal:

```bash
aws configure
```

Paste the access key ID, then the secret, then:
- Default region: `us-east-1`
- Default output: `json`

Verify:

```bash
aws sts get-caller-identity
```

Should print your account ID. If it errors, your credentials are wrong.

## 4. Install Docker and Terraform (5 min)

```bash
brew install --cask docker
brew install terraform
```

After Docker installs, **open the Docker Desktop app once** and let it finish setup. Then:

```bash
docker --version
terraform --version
```

Both should print version numbers.

## 5. Edit the email in Terraform vars (30 sec)

```bash
cd ~/Desktop/projectvik/klotho
open -a TextEdit infra/terraform.tfvars
```

Change `alert_email` to a real email if it isn't already correct. Save.

## 6. Test locally before deploying (5 min)

This catches problems before they cost you ECR upload time:

```bash
./scripts/local_test.sh
```

Should print health check, sample generation, and history. If it errors, paste the output before continuing.

## 7. Deploy to AWS (15 min first time)

```bash
./scripts/deploy.sh
```

What happens:

1. **Staging** — copies `out/best.pt` and `data/tokenizer.json` into the build context (~10 sec)
2. **First run only:** Terraform creates the ECR repo so we can push to it (~30 sec)
3. **ECR login** (~5 sec)
4. **Docker build** (5–10 min the first time — PyTorch is a big download. Subsequent builds are ~30s thanks to layer caching)
5. **Docker push** to ECR (~2 min for ~700 MB image on home internet)
6. **Terraform apply** — provisions Lambda, API Gateway, IAM, CloudWatch, budget alert (~1 min)
7. **Lambda update** — points the function at the new image (~30 sec)

When done, the script prints:

```
==> deployed.
    api: https://abc123xyz.execute-api.us-east-1.amazonaws.com
```

That's your live endpoint.

## 8. Test the live API (1 min)

```bash
API_URL=$(cd infra && terraform output -raw api_endpoint)

curl $API_URL/health
```

Should return `{"ok":true,"model_loaded":true,"model":{...}}`.

```bash
curl -X POST $API_URL/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ROMEO:","max_tokens":150,"temperature":0.8}'
```

Should return a Klotho completion. **First request will be slow (~10–15s)** because of cold start. Subsequent requests on the same warm container are fast.

```bash
curl $API_URL/history
```

Should return an array containing your previous generation.

## 9. Verify cost protection (1 min)

1. AWS Console → **Billing and Cost Management** (top right account dropdown → Billing)
2. Left sidebar → **Budgets**
3. You should see `klotho-monthly-budget` listed with a $1.00 limit and your email subscribed

If anything ever goes wrong, you'll get an email before you accrue a real charge.

## 10. Update later

Any time you change `serving/handler.py`, `model/model.py`, or any other source file, re-run:

```bash
./scripts/deploy.sh
```

It'll rebuild, push, and roll the Lambda function. Takes ~3 min total because Docker layer caching skips most of the work.

## 11. Tear down when done

After you've heard back from Pear:

```bash
./scripts/destroy.sh
```

Type `destroy` to confirm. This removes everything.

## Common gotchas

- **`docker buildx` errors on Apple Silicon** — make sure Docker Desktop is running. The `--platform linux/amd64` flag is intentional; we're cross-compiling because Lambda is x86_64.
- **`AccessDeniedException` from AWS** — your CLI credentials don't have admin rights. The default access key created from the root account should have full admin. Double-check `aws sts get-caller-identity`.
- **`ImagePullBackOff` style error from Lambda** — Lambda couldn't pull the image. Almost always means the architecture mismatched. Re-run `./scripts/deploy.sh` and confirm the build line says `--platform linux/amd64`.
- **Cold start takes 30+ seconds** — increase `lambda_memory_mb` in `infra/main.tf`. More memory = more CPU. 2048 is the default, try 3008 if needed. Cost still effectively zero at portfolio scale.
- **`terraform apply` fails with "BudgetServiceException"** — your AWS account is too new. Wait 24 hours after account creation before the Budgets API is fully enabled. Can also comment out the `aws_budgets_budget` resource and apply without it.
