#!/bin/bash
set -euo pipefail

#############################################
# Build and push Pluto training image to ECR
#############################################

AWS_REGION=${AWS_REGION:-us-west-2}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
REPO_NAME="pluto-train-kr"
IMAGE_TAG=${IMAGE_TAG:-$(date +%y%m%d%H%M)}
FULL_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "=== Building Pluto training image ==="
echo "Image: ${FULL_IMAGE}"
echo "Tag:   ${IMAGE_TAG}"

# Create ECR repo if not exists
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION}

# ECR login
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t ${FULL_IMAGE} ${SCRIPT_DIR}

# Push
docker push ${FULL_IMAGE}

echo "=== Done: ${FULL_IMAGE} ==="
