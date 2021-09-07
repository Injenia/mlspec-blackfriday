#!/bin/bash

set -e
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${current_dir}/config.sh

# remove artifact repository
gcloud artifacts repositories delete scann-${AI_PLATFORM_PREDICTION_REGION} --location=${AI_PLATFORM_PREDICTION_REGION} --quiet | true

# remove ai-platform model
gcloud ai-platform versions delete ${AI_PLATFORM_PREDICTION_VERSION} --model=${AI_PLATFORM_PREDICTION_MODEL} --region=${AI_PLATFORM_PREDICTION_REGION} --quiet | true
gcloud ai-platform models delete ${AI_PLATFORM_PREDICTION_MODEL} --region=${AI_PLATFORM_PREDICTION_REGION} --quiet | true