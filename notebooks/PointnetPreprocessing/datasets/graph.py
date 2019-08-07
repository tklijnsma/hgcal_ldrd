"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y.
"""

from collections import namedtuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y', 'simmatched'])
#from sparse_tensor import SpTensor



def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))


def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)


def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return Graph(**dict(f.items()))


def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]


#thanks Steve :-)
def draw_sample(X, Ri, Ro, y, out,
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]    
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    
    
    #if sim_list is None:    
        # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    #else:        
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    #    ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
    #    ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
        
     
    if out is not None:
        t = tqdm.tqdm(range(out.shape[0]))
        for j in t:       
            if y[j] and out[j]>0.5: 
                seg_args = dict(c='purple', alpha=0.2)
            elif y[j] and out[j]<0.5: 
                seg_args = dict(c='blue', alpha=0.2)
            elif out[j]>0.5:
                seg_args = dict(c='red', alpha=0.2)
            else:
                    continue #false edge

            ax0.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            ax1.plot([feats_o[j,1], feats_i[j,1]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else: continue
                
            ax0.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            ax1.plot([feats_o[j,1], feats_i[j,1]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
        
    # Adjust axes
    ax0.set_xlabel('$x$ [cm]')
    ax1.set_xlabel('$y$ [cm]')
    ax0.set_ylabel('$layer$ [arb]')
    ax1.set_ylabel('$layer$ [arb]')
    plt.tight_layout()
    return fig;