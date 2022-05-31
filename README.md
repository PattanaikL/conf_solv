# ConfSolv
Prediction of conformer solution free energy with deep learning

## Requirements

* python (version==3.7)
* rdkit (version==2022.03.1)
* pytorch (version==1.11.0)
* pytorch-geometric (version==2.0.4)
* pytorch-lightning (version==1.6.1)

## Installation

### Create the environment and activate
```
conda create -n ConfSolv python=3.7
conda activate ConfSolv
```

### Install pytorch
```
# CUDA 10.2
conda install pytorch cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch cudatoolkit=11.3 -c pytorch
# CPU only
conda install pytorch cpuonly -c pytorch
```

### Install required conda packages
```
conda install pyg -c pyg
conda install -c rdkit rdkit
conda install -c conda-forge einops
conda install -c conda-forge pytorch-lightning==1.6.1
conda install -c conda-forge ase
conda install -c anaconda sympy
```

### I had to do some special stuff to get jupyter working
```
conda install -c anaconda jupyter
conda install nb_conda
conda install nbconvert==6.4.3
```

### Finally, clone this repo
```
git clone https://github.com/PattanaikL/conf_solv
cd conf_solv
pip install -e .
```

### I used Neptune to keep track of experiments, and I thought it was useful to keep my experiments organized (optional, only for training)
```
conda install -c conda-forge neptune-client
```
