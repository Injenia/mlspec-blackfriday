#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${current_dir}/config.sh

curl -d '{"instances": [{"Gender":"M", "Age":"26-35", "Occupation":"0", "City_Category":"B", "Stay_In_Current_City_Years":"4+", "Marital_Status":"0"}]}' \
    -X POST https://${AI_PLATFORM_PREDICTION_REGION}-ml.googleapis.com/v1/projects/mlteam-ml-specialization-2021/models/${AI_PLATFORM_PREDICTION_MODEL}:predict?access_token\="$(gcloud auth application-default print-access-token)"