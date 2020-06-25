# HCAL specific model:

Clone the repo:

```
git clone https://github.com/tklijnsma/hgcal_ldrd.git
git checkout dev-hcal
```

The environment setup should be approximately as follows:

```
conda create -n hcal-env python=3.6
conda install -c pytorch magma-cuda100 pytorch
conda activate hcal-env
conda install -c pytorch magma-cuda100 pytorch
pip install --verbose --no-cache-dir torch_scatter
pip install --verbose --no-cache-dir torch_cluster
pip install --verbose --no-cache-dir torch-sparse torch-spline-conv
pip install torch-geometric
```

There are a few more packages, either look through [this conda environment dump](environment.yml) or just install them as you get import errors. I think doing `conda env create -f environment.yml` broke some things last time I tried it, but you could give it a go.

Make the directory structure for the data as follows:

```
my-hcal-dataset
my-hcal-dataset/raw
my-hcal-dataset/raw/0.npz
my-hcal-dataset/raw/1.npz
my-hcal-dataset/raw/2.npz
(.......)
my-hcal-dataset/raw/10007.npz
my-hcal-dataset/raw/10008.npz
my-hcal-dataset/raw/10009.npz

# Also make a copy with just 10 events in it for debugging
my-hcal-dataset-debug
my-hcal-dataset-debug/raw
my-hcal-dataset-debug/raw/0.npz
my-hcal-dataset-debug/raw/1.npz
my-hcal-dataset-debug/raw/2.npz
(.......)
my-hcal-dataset-debug/raw/8.npz
my-hcal-dataset-debug/raw/9.npz
```

Make some edits to [scripts/hcalscript.py](scripts/hcalscript.py#L147) to get the paths to the data correctly, they're now semi-hardcoded for my setup.

Then run the script:

```
cd hgcal-ldrd_script
python hcalscript.py
```

Once you run the script, it should create processed files:

```
my-hcal-dataset/processed/data_0.pt
my-hcal-dataset/processed/data_1.pt
(.......)
```

and the output should be created in `hgcal_ldrd/output`.


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

