# Training LSTM for SPT data

### Folder structure
```generate_trajectory.py``` - generate synthetic SPT dataset using andi-datasets package; mostly taked from the HNU team repo, from ANDI competition.  
```data_management.py``` - loading the dataset, preprocessing it, getting generators  
```hnu_replica.ipynb``` - main notebook with model training. It includes MLFlow for experiment tracking. To use it with mlflow, ```mlflow ui``` command needs to be ran.

### How to use
Prepare python environment:
```
python -m venv venv
source venv/bin/activate
pip install requirements.txt
```
Then launch the notebook ```hnu_replica.ipynb``` and run it cell-by-cell.

### Sources
Most of the code ideas taken from this repo:  
https://github.com/huangzih/AnDi-Challenge/tree/main
