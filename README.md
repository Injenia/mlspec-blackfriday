# Black Friday dataset

## Folders structure

### /dataset
Download `train.csv` and `test.csv` and place them under `data` folder.  
Kaggle dataset page is available [here](https://www.kaggle.com/abhisingh10p14/black-friday), you need a Kaggle account.

### /notebooks
Contains Jupyter notebooks to manage the dataset, main functionalities:
    - Dataset EDA
    - Dataset train / eval / test splitting
    - ML models scouting
In order to run the notebooks, install the packages using `/notebooks/notebooks_requirements.txt` 

### /trainer
Contains Python code for both local and AI Platform Jobs trainings. Code is structured as suggested in 
[GCP ai platform structure example](https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/training/tensorflow/census/tf-keras).
- Training requirements reported on `/requirements.txt` file  

### /deploy
Contains Bash helper scripts for model deploy/teardown, in particular:
- **config.sh**: configuration settings (actual model to be deployed, AI Platform model and version names, AI Platform region)
- **setup.sh**: creates a custom version of tf-serving-scann image that contains selected model, deploys the service as AI Platform model 
- **teardown.sh**: removes AI Platform version and model
- **invoke.sh**: example of inference invocation through HTTP API

For a new model deploy:

1. place the position of the exported model in `MODELDIR` variable in **config.sh**, optionally use `AI_PLATFORM_PREDICTION_VERSION` and `AI_PLATFORM_PREDICTION_MODEL` variables to change model name and version
2. call **setup.sh**
3. test deployed model with **invoke.sh** script


### /scripts
Contains Bash helper scripts for local/cloud based training and hyperparameter tuning.