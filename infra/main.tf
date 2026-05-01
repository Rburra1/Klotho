terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "klotho"
      Environment = "portfolio"
      ManagedBy   = "terraform"
      Owner       = "rithvik-burra"
    }
  }
}

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name; used as a prefix for resource naming."
  type        = string
  default     = "klotho"
}

variable "image_tag" {
  description = "ECR image tag to deploy. Set to the digest after pushing."
  type        = string
  default     = "latest"
}

variable "lambda_memory_mb" {
  description = "Lambda memory in MB. More memory also gives proportionally more CPU."
  type        = number
  default     = 2048
}

variable "lambda_timeout_seconds" {
  description = "Lambda invocation timeout."
  type        = number
  default     = 30
}

# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

locals {
  name        = var.project
  account_id  = data.aws_caller_identity.current.account_id
  ecr_repo_uri = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${local.name}"
}

# ----------------------------------------------------------------------------
# ECR - container image registry for the Lambda image
# ----------------------------------------------------------------------------

resource "aws_ecr_repository" "klotho" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # Make 'terraform destroy' actually remove the repo even if it still has images.
  force_delete = true
}

# Keep only the most recent 5 images; deletes older ones automatically to stay
# under the ECR free tier (500 MB/month for first 12 months).
resource "aws_ecr_lifecycle_policy" "klotho" {
  repository = aws_ecr_repository.klotho.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images only"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      }
    ]
  })
}

# ----------------------------------------------------------------------------
# IAM role for Lambda
# ----------------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "${local.name}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

# Attach the AWS managed policy that allows Lambda to write to CloudWatch Logs.
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ----------------------------------------------------------------------------
# CloudWatch log group with explicit retention so logs don't accumulate forever
# ----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "klotho" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 7
}

# ----------------------------------------------------------------------------
# Lambda function (container image)
# ----------------------------------------------------------------------------

resource "aws_lambda_function" "klotho" {
  function_name = local.name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.klotho.repository_url}:${var.image_tag}"

  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds

  # Reserved concurrency caps simultaneous executions and protects against
  # runaway invocation costs. 5 is plenty for a portfolio API.
  reserved_concurrent_executions = 5

  environment {
    variables = {
      ATHENA_MODEL_DIR = "/var/task/model_artifacts"
      ATHENA_DB_PATH   = "/tmp/klotho.db"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_logs,
    aws_cloudwatch_log_group.klotho,
  ]
}

# ----------------------------------------------------------------------------
# API Gateway HTTP API (cheaper and simpler than REST API for this case)
# ----------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "klotho" {
  name          = "${local.name}-http-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_integration" "klotho_lambda" {
  api_id                 = aws_apigatewayv2_api.klotho.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.klotho.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 29000
}

# Routes
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.klotho.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.klotho_lambda.id}"
}

resource "aws_apigatewayv2_route" "generate" {
  api_id    = aws_apigatewayv2_api.klotho.id
  route_key = "POST /generate"
  target    = "integrations/${aws_apigatewayv2_integration.klotho_lambda.id}"
}

resource "aws_apigatewayv2_route" "history" {
  api_id    = aws_apigatewayv2_api.klotho.id
  route_key = "GET /history"
  target    = "integrations/${aws_apigatewayv2_integration.klotho_lambda.id}"
}

# Default stage with auto-deploy
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.klotho.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 10
    throttling_rate_limit  = 5
  }
}

# Permission for API Gateway to invoke the Lambda
resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.klotho.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.klotho.execution_arn}/*/*"
}

# ----------------------------------------------------------------------------
# Budget alert - belt and suspenders for cost protection
# ----------------------------------------------------------------------------

resource "aws_budgets_budget" "monthly" {
  name              = "${local.name}-monthly-budget"
  budget_type       = "COST"
  limit_amount      = "1.00"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}

variable "alert_email" {
  description = "Email to notify when monthly spend exceeds $1."
  type        = string
}

# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

output "api_endpoint" {
  description = "Public HTTPS endpoint for the Klotho API."
  value       = aws_apigatewayv2_api.klotho.api_endpoint
}

output "ecr_repository_url" {
  description = "ECR repo URL. Push container images here."
  value       = aws_ecr_repository.klotho.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.klotho.function_name
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.klotho.name
}
