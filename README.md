# Jit EdgeNetWithCategories

Working environment (24 July 2020):

```
conda install cudatoolkit=10.2
conda install -c pytorch pytorch=1.5.0

export CUDA="cu102"
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html

git clone https://github.com/rusty1s/pytorch_geometric.git -b 1.6.0
pip install -e pytorch_geometric/
```

`pip install torch-geometric` might work again in the nearby future if the pypi packaging is fixed.

Jit EdgeNetWithCategories with [scripts/jit_edgenetwithcategories.py](scripts/jit_edgenetwithcategories.py) .


# hgcal_ldrd
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

