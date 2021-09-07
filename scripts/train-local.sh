#!/bin/bash
set -v

echo "Training Cloud ML model"

DATE=$(date '+%Y%m%d_%H%M%S')

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=bf_$(date +%Y%m%d_%H%M%S)

# JOB_DIR: the output directory
JOB_DIR=gs://mlteam-ml-specialization-2021-blackfriday/keras-job-dir # TODO Change BUCKET_NAME to your bucket name

gcloud ai-platform local train \
  --package-path trainer/ \
  --module-name trainer.task \
  --job-dir "${JOB_DIR}"
