#!/bin/bash

set -e
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${current_dir}/config.sh

# downloading model
rm -rf ${TEMPDIR} | true
mkdir -p ${TEMPDIR}/Scann/01
gsutil -m cp -r ${MODELDIR}/* ${TEMPDIR}/Scann/01/

# building image
cd ${TEMPDIR}
docker pull google/tf-serving-scann
docker run -d --name serving_base google/tf-serving-scann
docker cp Scann serving_base:/models/Scann
docker commit --change "ENV MODEL_NAME Scann" serving_base ${AI_PLATFORM_PREDICTION_VERSION}
docker kill serving_base

# createing artifactory repository (if not present)
gcloud artifacts repositories create scann-${AI_PLATFORM_PREDICTION_REGION} \
--repository-format=docker --location=${AI_PLATFORM_PREDICTION_REGION} | true

# pushing image to artifactory
gcloud auth configure-docker ${AI_PLATFORM_PREDICTION_REGION}-docker.pkg.dev
docker tag ${AI_PLATFORM_PREDICTION_VERSION} ${AI_PLATFORM_PREDICTION_REGION}-docker.pkg.dev/mlteam-ml-specialization-2021/scann-europe-west1/${AI_PLATFORM_PREDICTION_VERSION}
docker push ${AI_PLATFORM_PREDICTION_REGION}-docker.pkg.dev/mlteam-ml-specialization-2021/scann-europe-west1/${AI_PLATFORM_PREDICTION_VERSION}

# deploy the image to ai platform prediction
gcloud ai-platform models create ${AI_PLATFORM_PREDICTION_MODEL} --region=${AI_PLATFORM_PREDICTION_REGION}  --enable-logging | true
gcloud beta ai-platform versions create ${AI_PLATFORM_PREDICTION_VERSION} \
  --region=${AI_PLATFORM_PREDICTION_REGION} \
  --model=${AI_PLATFORM_PREDICTION_MODEL} \
  --machine-type=n1-standard-4 \
  --image=${AI_PLATFORM_PREDICTION_REGION}-docker.pkg.dev/mlteam-ml-specialization-2021/scann-europe-west1/${AI_PLATFORM_PREDICTION_VERSION} \
  --ports=8501 \
  --health-route=/v1/models/Scann \
  --predict-route=/v1/models/Scann:predict
gcloud ai-platform versions set-default ${AI_PLATFORM_PREDICTION_VERSION} --model=${AI_PLATFORM_PREDICTION_MODEL} --region=${AI_PLATFORM_PREDICTION_REGION}
  
# local cleanup
rm -rf ${TEMPDIR} | true
docker rm /serving_base
docker image rm -f ${AI_PLATFORM_PREDICTION_VERSION}
docker image rm -f ${AI_PLATFORM_PREDICTION_REGION}-docker.pkg.dev/mlteam-ml-specialization-2021/scann-europe-west1/${AI_PLATFORM_PREDICTION_VERSION}
docker image rm -f google/tf-serving-scann

# list deployed models
echo "listing deployed models in region ${AI_PLATFORM_PREDICTION_REGION}"
gcloud ai-platform models list --region=${AI_PLATFORM_PREDICTION_REGION}

