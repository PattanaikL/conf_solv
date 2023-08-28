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
