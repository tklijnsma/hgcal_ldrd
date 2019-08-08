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
def draw_sample_point(X, y, out): 
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))

    truepos = np.logical_and(y, out)
    falseneg = np.logical_and(np.logical_not(out), y)
    falsepos = np.logical_and(out, np.logical_not(y))
    trueneg = np.logical_and(np.logical_not(y), np.logical_not(out))
    
    assert(y.shape[0] == truepos.sum() + falseneg.sum() + falsepos.sum() + trueneg.sum())
    
    ax0.scatter(X[truepos][:,0], X[truepos][:,2], c='green',alpha=0.7)
    ax1.scatter(X[truepos][:,1], X[truepos][:,2], c='green',alpha=0.7)    
    ax0.scatter(X[falseneg][:,0], X[falseneg][:,2], c='red',alpha=0.7)
    ax1.scatter(X[falseneg][:,1], X[falseneg][:,2], c='red',alpha=0.7)
    ax0.scatter(X[falsepos][:,0], X[falsepos][:,2], c='blue',alpha=0.2)
    ax1.scatter(X[falsepos][:,1], X[falsepos][:,2], c='blue',alpha=0.2)    
    ax0.scatter(X[trueneg][:,0], X[trueneg][:,2], c='k',alpha=0.02)
    ax1.scatter(X[trueneg][:,1], X[trueneg][:,2], c='k',alpha=0.02)
    
        
    # Adjust axes
    ax0.set_xlabel('$x$ [cm]')
    ax1.set_xlabel('$y$ [cm]')
    ax0.set_ylabel('$z$ [cm]')
    ax1.set_ylabel('$z$ [cm]')
    plt.tight_layout()
    return fig;