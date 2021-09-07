#!/bin/bash
set -v

echo "Training Cloud ML model"

DATE=$(date '+%Y%m%d_%H%M%S')

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=bf_$(date +%Y%m%d_%H%M%S)

# JOB_DIR: the output directory
JOB_DIR=gs://mlteam-ml-specialization-2021-blackfriday/aiplatform_jobs/${JOB_NAME}

# REGION: select a region from https://cloud.google.com/ai-platform/training/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=europe-west1
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.4
TRAIN_STEPS=10
EVAL_STEPS=10
BATCH_SIZE=30000 #1 epoch -> 3 steps
NUM_EPOCHS=1000

current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd ${current_dir}"/.."

config_file="${current_dir}/config.yaml"

gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --package-path trainer/ \
  --module-name trainer.task \
  --region ${REGION} \
  --python-version ${PYTHON_VERSION} \
  --runtime-version ${RUNTIME_VERSION} \
  --job-dir "${JOB_DIR}" \
  --config "${config_file}" \
  --stream-logs -- \
  --job-dir="${JOB_DIR}" \
  --num-epochs=${NUM_EPOCHS} \
  --batch-size=${BATCH_SIZE}

#  --stream-logs -- \
#  --train-steps=${TRAIN_STEPS} \
#  --eval-steps=${EVAL_STEPS} \
