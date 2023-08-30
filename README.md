# ConfSolv
Prediction of conformer free energies in solution with deep learning.

## Key Requirements

* python (version==3.7)
* rdkit (version==2020.09.1.0)
* pytorch (version==1.11.0)
* pytorch-geometric (version==2.0.4)
* pytorch-lightning (version==1.6.1)

## Installation

### Creating the environment
To ensure all the appropriate packages and versions are installed, it is strongly recommended to use the environment.yml file:
``` 
conda env create -f environment.yml
conda activate ConfSolv
```
The environment has all necessary GPU and Jupyter support.  


### Clone the main repository and install
```
git clone https://github.com/PattanaikL/conf_solv
cd conf_solv
pip install -e .
```
## Training the Model 
An example submission script for training a DimeNet++ model is provided in the submission_scripts folder. This submission script uses SLURM and is submitted with: 
```
sbatch train_dimenet.sh
```
The script should be easily adaptable for other schedulers. To train the model locally, one can call train.py directly with the necessary arguments:
```
python conf_solv/train.py args-list  
```

## Using the Model
Previously trained models are found in the sample_trained_models folder. An example Jupyter Notebook on how to load and use these models is provided in the inference folder. 
For making predictions on a large number of solutes and solvents using a single trained model, the predict.sh submission script in the submission_scripts folder can be used. As with the training script, this is submitted with:
```
sbatch predict.sh
```
Predictions are generated for each solute conformer in each of the available solvents.
