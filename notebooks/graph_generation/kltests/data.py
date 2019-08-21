import uproot
import awkward
import numpy as np
import os, sys
import os.path as osp
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm_notebook as tqdm
import xgboost as xgb

sys.path.append(osp.abspath('..'))

from graph import (
    SparseGraph, make_sparse_graph,
    save_graph, save_graphs, load_graph,
    load_graphs, make_sparse_graph, graph_from_sparse,
    draw_sample_validation, draw_sample3d
    )

from preprocessing import (
    make_graph_xy, make_graph_etaphi, make_graph_knn, make_graph_kdtree
    )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import importlib
from time import strftime
datestr = strftime('%b%d')

def reload():
    importlib.reload(sys.modules[__name__])


class DataContainer(object):
    """docstring for DataContainer"""
    def __init__(self):
        super(DataContainer, self).__init__()
        
        print('Initializing data container')

        self.fname = '/home/thomas/data/pions/partGun_PDGid211_x1000_E5.0To1000.0_NTUP_1.root'
        self.layer_norm = 150
        self.verbose = False

        self.read_ntup()


    def p(self, *args, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose: print(*args)


    def read_ntup(self):

        self.p('Reading data {0}'.format(self.fname))
        self.test = uproot.open(self.fname)['ana']['hgc']

        self.arrays = self.test.arrays([b'simcluster_hits_indices'])
        self.rechit = self.test.arrays([b'rechit_x',b'rechit_y', b'rechit_z', b'rechit_layer',b'rechit_time',b'rechit_energy'])

        self.n_events = self.test.numentries

        self.sim_indices         = awkward.fromiter(self.arrays[b'simcluster_hits_indices'])
        self.valid_sim_indices   = self.sim_indices[self.sim_indices > -1]

        if self.verbose:
            for key in self.arrays.keys():
                self.p('self.arrays[{0}]: {1}'.format(key, self.arrays[key].shape))
            for key in self.rechit.keys():
                self.p('self.rechit[{0}]: {1}'.format(key, self.rechit[key].shape))

            ievt = 5
            self.p('\nFor ievt = {0}'.format(ievt))
            for key in self.arrays.keys():
                self.p('self.arrays[{0}][{2}]: {1}'.format(key, len(self.arrays[key][ievt]), ievt))
            for key in self.rechit.keys():
                self.p('self.rechit[{0}][{2}]: {1}'.format(key, self.rechit[key][ievt].shape, ievt))


            self.p('self.sim_indices:', self.sim_indices.shape)
            self.p('self.sim_indices[{0}]:'.format(ievt), self.sim_indices[ievt].shape)
            self.p('self.sim_indices[{0}][0]:'.format(ievt), self.sim_indices[ievt][0].shape)
            self.p('')
            self.p('self.valid_sim_indices:', self.valid_sim_indices.shape)
            self.p('self.valid_sim_indices[{0}]:'.format(ievt), self.valid_sim_indices[ievt].shape)
            self.p('self.valid_sim_indices[{0}][0]:'.format(ievt), self.valid_sim_indices[ievt][0].shape)


    def get_features(self, ievt, mask):
        x      = rechit_x[ievt][mask]
        y      = rechit_y[ievt][mask]
        layer  = rechit_layer[ievt][mask]
        time   = rechit_time[ievt][mask]
        energy = rechit_energy[ievt][mask]    
        return np.stack((x,y,layer,time,energy)).T


data = DataContainer()