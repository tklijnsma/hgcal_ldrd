import uproot
import awkward
import numpy as np
import os, sys
import os.path as osp
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm_notebook as tqdm
import xgboost as xgb

import sys
import numpy as np
import awkward
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, find, tril, triu
from sklearn.neighbors import NearestNeighbors
from itertools import tee


sys.path.append(osp.abspath('..'))

from graph import Graph

from graph import (
    SparseGraph, make_sparse_graph,
    save_graph, save_graphs, load_graph,
    load_graphs, make_sparse_graph, graph_from_sparse,
    draw_sample_validation, draw_sample3d
    )

import preprocessing
from preprocessing import (
    make_graph_xy, make_graph_etaphi, make_graph_kdtree,
    # make_graph_knn,
    make_graph_knn_old
    )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %load_ext autoreload
# %autoreload 2


import importlib
from time import strftime
datestr = strftime('%b%d')

def reload():
    importlib.reload(sys.modules[__name__])
    importlib.reload(preprocessing)
    from preprocessing import (
        make_graph_xy, make_graph_etaphi, make_graph_knn, make_graph_kdtree, make_graph_knn_old
        )

from data import data




class NoisePredictor(object):
    """docstring for NoisePredictor"""
    def __init__(self, path_to_model):
        super(NoisePredictor, self).__init__()
        self.path_to_model = path_to_model
        self.bst = xgb.Booster()
        self.bst.load_model(path_to_model)
        self.cut_point = 0.0334665 

    def is_noise(self, hit):
        return bst.predict(xgb.DMatrix(hit)) > self.cut_point



class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self):
        super(Preprocessor, self).__init__()

        self.preprocessing_algo = make_graph_xy
        self.grouping_algo = make_graph_knn
        
        self.preprocessing_args = dict(k=4)

        self.plotdir = 'plots_{0}'.format(datestr)
        self.data = data

        self._has_noise_predictor = False



    def setup_noise_predictor(self):
        if self._has_noise_predictor: return
        self.noise_predictor = NoisePredictor(
            '/home/thomas/inferhgcal1/hgcal_ldrd/notebooks/graph_generation/kltests/max_depth_5_num_round_8000.model'
            )
        self._has_noise_predictor = True


    def test_noise_filtering(self):
        self.setup_noise_predictor()
        
        # n_test = self.data.n_events
        n_test = 10

        n_real = 0
        n_noise = 0

        false_positives = 0
        false_negatives = 0
        correct_positives = 0
        correct_negatives = 0

        for i in range(n_test):

            # There are two pions per event, just pick whichever one is at index 0
            hit = self.data.rechit[i][0]

            is_real_sim = self.data.valid_sim_indices[i][0] == True
            is_noise_sim = not(is_real_sim)

            is_noise_pred = self.noise_predictor.is_noise(hit)
            is_real_pred = not(is_noise_pred)


            if is_real_sim:
                n_real += 1
                if is_noise_pred:
                    false_negatives += 1
                else:
                    correct_positives += 1
            else:
                n_noise += 1
                if is_noise_pred:
                    correct_negatives += 1
                else:
                    false_positives += 1


            # Avoid division issues
            n_real = float(n_real)
            n_noise = float(n_noise)

        print('% correct real hits / all hits:', correct_positives/n_real)
        print('% incorrect real hits / all hits:', false_negatives/n_real)


        print('% correct noise hits / all hits:', correct_negatives/n_noise)
        print('% incorrect noise hits / all hits:', false_positives/n_noise)



    def edge_efficiency(self, graph):

        pass


    def loop_edge_efficiency_knn(self, k):

        for ievt in range(self.data.n_events):

            graph = self.preprocessing_algo(
                self.data.rechit,
                self.data.valid_sim_indices,
                ievt               = ievt,
                mask               = self.data.rechit[b'rechit_z'][ievt] > 0,
                layered_norm       = self.data.layer_norm,
                algo               = self.grouping_algo,
                preprocessing_args = self.preprocessing_args
                )




            if ievt > 10: break


    def get_features(self, ievt, mask):
        x      = rechit_x[ievt][mask]
        y      = rechit_y[ievt][mask]
        layer  = rechit_layer[ievt][mask]
        time   = rechit_time[ievt][mask]
        energy = rechit_energy[ievt][mask]    
        return np.stack((x,y,layer,time,energy)).T


    def plot3d(self, graph, plotname='graph3d'):
        fig = plt.figure()
        ax = Axes3D(fig)
        
        x = graph.X[:,0]
        y = graph.X[:,1]
        z = graph.X[:,2]
        
        print('x:', x)
        print('y:', y)
        print('z:', z)
        
        ax.scatter(x, y, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.scatter([0], [0], [0], c='r')    

        # plt.show()

        if not osp.isdir(self.plotdir): os.makedirs(self.plotdir)
        out = osp.join(self.plotdir, plotname + '.png' if not plotname.endswith('.png') else plotname)
        plt.savefig(out)



    def test_graph(self):
        print('Running test_graph')

        print(self.data.sim_indices[5].shape)
        print(self.data.sim_indices[5][0])
        print(self.data.sim_indices[5][0].shape)
        print(self.data.sim_indices[5][1])
        print(self.data.sim_indices[5][1].shape)

        # return

        ievt = 5
        graph = self.preprocessing_algo(
            self.data.rechit,
            self.data.valid_sim_indices,
            ievt               = ievt,
            mask               = self.data.rechit[b'rechit_z'][ievt] > 0,
            layered_norm       = self.data.layer_norm,
            algo               = self.grouping_algo,
            preprocessing_args = self.preprocessing_args
            )

        print(graph.Ro.shape)
        print(graph.Ri.shape)

        print(graph.Ro[:4])
        print('---------')
        print(graph.Ri[:4])

        print(graph.simmatched)



    def test_plot3d(self):
        print('Running test plots 3d')

        # n_events = self.n_events
        n_events = 5

        for ievt in tqdm(range(n_events),desc='events processed'):
            #make input graphs
            
            pos_graph = self.preprocessing_algo(
                self.data.rechit,
                self.data.valid_sim_indices,
                ievt               = ievt,
                mask               = self.data.rechit[b'rechit_z'][ievt] > 0,
                layered_norm       = self.data.layer_norm,
                algo               = self.grouping_algo,
                preprocessing_args = self.preprocessing_args
                )
            
            neg_graph = self.preprocessing_algo(
                self.data.rechit,
                self.data.valid_sim_indices,
                ievt               = ievt,
                mask               = self.data.rechit[b'rechit_z'][ievt] < 0,
                layered_norm       = self.data.layer_norm,
                algo               = self.grouping_algo,
                preprocessing_args = self.preprocessing_args
                )

            if ievt < 5:
                print('-' * 40 + '\nievt = ' + str(ievt))
                print('n hits:', len(pos_graph.X[:,0]))
                self.plot3d(pos_graph, 'graph3d_{0}'.format(ievt))
            
            # #write the graph and truth graph out
            # outbase = fname.split('/')[-1].replace('.root','')
            # if not os.path.exists('./' + outbase):
            #     os.makedirs('./' + outbase)
            
            # graph = make_sparse_graph(*pos_graph)
            # save_graph(graph, '%s/%s_hgcal_graph_pos_evt%d.npz'%(outbase,outbase,ievt))
                
            # graph = make_sparse_graph(*neg_graph)
            # save_graph(graph, '%s/%s_hgcal_graph_neg_evt%d.npz'%(outbase,outbase,ievt))


    def test_draw_xz(self):

        ievt = 0

        g1 = self.preprocessing_algo(
            self.data.rechit,
            self.data.valid_sim_indices,
            ievt               = ievt,
            mask               = self.data.rechit[b'rechit_z'][ievt] > 0,
            layered_norm       = self.data.layer_norm,
            algo               = make_graph_knn,
            preprocessing_args = self.preprocessing_args
            )

        # return

        # g2 = self.preprocessing_algo(
        #     self.data.rechit,
        #     self.data.valid_sim_indices,
        #     ievt               = ievt,
        #     mask               = self.data.rechit[b'rechit_z'][ievt] > 0,
        #     layered_norm       = self.data.layer_norm,
        #     algo               = make_graph_knn_old,
        #     preprocessing_args = self.preprocessing_args
        #     )


        print(g1.X.shape, g1.Ri.shape, g1.Ro.shape, g1.y.shape, g1.simmatched.shape)
        # print(g2.X.shape, g2.Ri.shape, g2.Ro.shape, g2.y.shape, g2.simmatched.shape)

        draw_sample_validation(
            g1.X, g1.Ri, g1.Ro, g1.y,
            sim_list = g1.simmatched, 
            skip_false_edges = False
            )



def make_graph_knn(coords, layers, sim_indices, k):
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(coords)
    nbrs_sm = nbrs.kneighbors_graph(coords, k)

    nbrs_sm = symmetrize_csr(nbrs_sm)

    nbrs_sm.setdiag(0) #remove self-loop edges
    nbrs_sm.eliminate_zeros() 

    Ri, Ro = connectivity_mat_to_RiRo(nbrs_sm)

    print('\nIn make_graph_knn')

    print(np.array(nbrs_sm.nonzero()).T.shape)
    print(np.array(triu(nbrs_sm).nonzero()).T.shape)
    print(sim_indices.shape)

    print(np.array(triu(nbrs_sm).nonzero()).T)

    print(
        np.isin(
            np.array(triu(nbrs_sm).nonzero()).T,
            sim_indices
            )
        )
    # print(sim_indices)

    # print('here')
    # print(nbrs_sm.shape)
    # print(nbrs_sm)

    
    y = (
        np.isin(
            np.array(triu(nbrs_sm).nonzero()).T,
            sim_indices
            ).astype(np.int8).sum(axis=-1)
        == 2
        )

    y2 = (
        np.isin(
            np.array(nbrs_sm.nonzero()).T,
            sim_indices
            ).astype(np.int8).sum(axis=-1)
        == 2
        )

    print(y.shape)
    print(y)
    print(y2.shape)
    print(y2)

    return Ri, Ro, y

def symmetrize_csr(M, use_lower_triangle=False):
    triangle = triu(M) if not(use_lower_triangle) else tril(M)
    return triangle + triangle.T

def connectivity_mat_to_RiRo(M, verbose=False):
    n_nodes = M.shape[0]

    if verbose: print('-'*50)
    if verbose: print(M.toarray())

    Mi = tril(M)
    Mo = triu(M)

    if verbose: print('-'*50)
    if verbose: print(Mi.toarray())
    if verbose: print('-'*50)
    if verbose: print(Mo.toarray())

    # Ri
    rows, cols = Mi.nonzero()
    n_edges = rows.shape[0]
    if verbose: print('-'*50)
    if verbose: print(rows)
    if verbose: print(cols)

    ones         = np.ones(n_edges)
    edge_numbers = np.arange(n_edges)

    Ri = csr_matrix(
        (ones, (rows,edge_numbers)),
        shape = (n_nodes,n_edges),
        dtype = int
        )
    if verbose: print(Ri.toarray())

    # Ro
    rows, cols = Mo.nonzero()
    n_edges = rows.shape[0]
    if verbose: print('-'*50)
    if verbose: print(rows)
    if verbose: print(cols)

    ones         = np.ones(n_edges)
    edge_numbers = np.arange(n_edges)

    Ro = csr_matrix(
        (ones, (rows,edge_numbers)),
        shape = (n_nodes,n_edges),
        dtype = int
        )
    if verbose: print(Ro.toarray())

    return Ri, Ro


p = Preprocessor()