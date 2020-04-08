# PVCNN for HGCAL


## Installation instructions:

Setup the environment (this may take a while):

```
wget https://raw.githubusercontent.com/tklijnsma/hgcal_ldrd/dev-pvcnn/environment.yml
conda env create -f environment.yml
conda activate pvcnn
```

Clone this repository, and the main pvcnn repository:

```
git clone https://github.com/tklijnsma/hgcal_ldrd.git -b dev-pvcnn
git clone https://github.com/mit-han-lab/pvcnn.git
```

The script by default assumes there is a directory `data` at the same level. Send me a message to obtain training data.


## Usage

```
python hgcal_ldrd/scripts/pvcnn-script.py
```

______________________________

# Main hgcal_ldrd instructions

Code repository for HGCal LDRD

You will need to:
```
conda create --name hgcal-env python=3.6
conda activate hgcal-env
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install pandas matplotlib jupyter nbconvert==5.4.1
conda install -c conda-forge tqdm
pip install uproot scipy sklearn --user
pip install torch-scatter torch-sparse
pip install networkx
```
(or replace pip with the corresponding conda installation for safer compatibility)

and install pytorch geometric according to the instructions here:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

