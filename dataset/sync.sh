#!/bin/bash

# ---- ---- ---- ---- ---- ---- ---- ---- 
# Script to copy "parsed" dataset to gcs 
# ---- ---- ---- ---- ---- ---- ---- ---- 
set -e

GCS_NAME=${1:-"mlteam-ml-specialization-2021-blackfriday"}
GCS_DATASET_ROOT=${2:-"dataset"}

SUBJECT="sync-bf-gcs"
cd "$(dirname "$0")"
LOCAL_FOLDER="./parsed/"

# --- Locks -------------------------------------------------------
LOCK_FILE=/tmp/$SUBJECT.lock
if [ -f "$LOCK_FILE" ]; then
   echo "Script is already running"
   exit
fi

trap "rm -f $LOCK_FILE" EXIT
touch $LOCK_FILE

# --- Body --------------------------------------------------------
GCS_PATH="gs://$GCS_NAME/$GCS_DATASET_ROOT/"
echo "Copying './parsed/' folder to GCS path '$GCS_PATH'"
gsutil -m cp -r -n $LOCAL_FOLDER $GCS_PATH # -n = skip if already exist
echo "> Copy completed!"
# -----------------------------------------------------------------