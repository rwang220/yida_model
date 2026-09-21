#!/usr/bin/env bash
set -euo pipefail

# One-time: create S3 + CloudFront, then upload docs/index.html.
# Requires AWS CLI v2, logged into a global AWS account (not aws.cn).
#
#   chmod +x infra/bootstrap-website.sh
#   ./infra/bootstrap-website.sh
#
# Optional:
#   STACK_NAME=yida-model-website BUCKET_NAME=your-unique-bucket ./infra/bootstrap-website.sh
#
# Mainland China: this stack uses global CloudFront. It is not an ICP-filed
# China site. Add a China CDN later if that audience matters.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STACK_NAME="${STACK_NAME:-yida-model-website}"
BUCKET_NAME="${BUCKET_NAME:-yida-model-website-rwang220}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-east-1}}"
TEMPLATE="${ROOT}/infra/website-cloudfront.yaml"
SITE_DIR="${ROOT}/docs"
SITE_FILE="${SITE_DIR}/index.html"

if ! command -v aws >/dev/null 2>&1; then
  echo "Install AWS CLI v2 first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

if [[ ! -f "$SITE_FILE" ]]; then
  echo "Missing $SITE_FILE"
  exit 1
fi

echo "Identity:"
aws sts get-caller-identity --output table

echo "Deploying stack $STACK_NAME in $REGION ..."
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --parameter-overrides "BucketName=$BUCKET_NAME" \
  --region "$REGION" \
  --no-fail-on-empty-changeset

BUCKET="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)"
DIST_ID="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='DistributionId'].OutputValue" --output text)"
URL="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='WebsiteURL'].OutputValue" --output text)"

echo "Uploading site to s3://$BUCKET/"
aws s3 sync "$SITE_DIR" "s3://$BUCKET/" \
  --region "$REGION" \
  --exclude ".nojekyll" \
  --exclude "*.md" \
  --cache-control "public, max-age=86400"
aws s3 cp "$SITE_FILE" "s3://$BUCKET/index.html" \
  --region "$REGION" \
  --content-type "text/html; charset=utf-8" \
  --cache-control "public, max-age=300"
aws s3 cp "$SITE_DIR/assets/site.css" "s3://$BUCKET/assets/site.css" \
  --region "$REGION" \
  --content-type "text/css; charset=utf-8" \
  --cache-control "public, max-age=86400"

echo "Invalidating CloudFront $DIST_ID"
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
POLICY="${ROOT}/infra/github-deploy-policy.json"
TMP_POLICY="$(mktemp)"
sed -e "s/REPLACE_BUCKET_NAME/${BUCKET}/g" \
    -e "s/REPLACE_ACCOUNT_ID/${ACCOUNT_ID}/g" \
    -e "s/REPLACE_DISTRIBUTION_ID/${DIST_ID}/g" \
    "$POLICY" > "$TMP_POLICY"

echo
echo "Website URL (wait 5–15 minutes on first CloudFront deploy):"
echo "  $URL"
echo
echo "GitHub repository secrets / variables:"
echo "  AWS_ACCESS_KEY_ID           IAM user access key"
echo "  AWS_SECRET_ACCESS_KEY       IAM user secret"
echo "  WEBSITE_S3_BUCKET           $BUCKET"
echo "  CLOUDFRONT_DISTRIBUTION_ID  $DIST_ID"
echo "  Repository variable AWS_WEBSITE_DEPLOY=true  (enables deploy on push)"
echo
echo "Attach this policy to the deploy IAM user:"
cat "$TMP_POLICY"
rm -f "$TMP_POLICY"
