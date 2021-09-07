# Black Friday dataset

## Folders structure

### /dataset
Download `train.csv` and `test.csv` and place them under `data` folder.  
Kaggle dataset page is available [here](https://www.kaggle.com/abhisingh10p14/black-friday), you need a Kaggle account.

### /notebooks
Contain all the notebook to manage the dataset, main functionalities:
    - Dataset EDA
    - Dataset train / eval / test splitting
    - ML models scouting
In order to run the notebooks, install the packages using `/notebooks/notebooks_requirements.txt` 

### /trainer
Contain all the code for run AI Platform Jobs trainings.
- Training requirements reported on `/requirements.txt` file  

**Resources**
- GCP ai platform structure example [GitHub](https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/training/tensorflow/census/tf-keras)