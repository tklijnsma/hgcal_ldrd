# hgcal_ldrd
Code repository for HGCal LDRD

You will need to:
```
conda create --name hgcal-env python=3.6
conda activate hgcal-env
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install pandas matplotlib jupyter nbconvert==5.4.1
conda install -c conda-forge tqdm
pip install uproot scipy sklearn --user
pip install networkx
```

and install pytorch geometric according to the instructions here:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
