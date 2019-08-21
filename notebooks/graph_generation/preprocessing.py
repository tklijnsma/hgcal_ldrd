import sys
import numpy as np
import awkward
from graph import Graph
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, find, tril, triu
from sklearn.neighbors import NearestNeighbors
from itertools import tee


def symmetrize_np(a, use_lower_triangle=False):
    n = a.shape[0]
    for i in range(n):
        for j in (range(i) if use_lower_triangle else range(i,n)):
            a[j,i] = a[i,j]

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


def csr_is_symmetric(c):
    return (c != c.T).nnz == 0


def make_graph_kdtree(coords,layers,sim_indices,r):
    #setup kd tree for fast processing
    the_tree = cKDTree(coords)
    
    #define the pre-processing (all layer-adjacent hits in ball R < r)
    #and build a sparse matrix representation, then blow it up 
    #to the full R_in / R_out definiton
    pairs = the_tree.query_pairs(r=r,output_type='ndarray')
    pairs = pairs[np.argsort(pairs[:,0])]
    first,second = pairs[:,0],pairs[:,1]  
    #selected index pair list that we label as connected
    #pairs_sel  = pairs[( (np.abs(layers[(second,)]-layers[(first,)]) <= 1)  )]
    neighbour_counts = np.unique(pairs[:,0], return_counts=True)[1]
    neighbour_counts = np.repeat(neighbour_counts, neighbour_counts)
    pairs_sel  = pairs[(np.abs(layers[(second,)]-layers[(first,)]) <= 1) | (neighbour_counts == 1)]
    #pairs_sel  = pairs
    data_sel = np.ones(pairs_sel.shape[0])
    
    #prepare the input and output matrices (already need to store sparse)
    r_shape = (coords.shape[0],pairs.shape[0])
    eye_edges = np.arange(pairs_sel.shape[0])
    
    R_i = csr_matrix((data_sel,(pairs_sel[:,1],eye_edges)),r_shape,dtype=np.uint8)
    R_o = csr_matrix((data_sel,(pairs_sel[:,0],eye_edges)),r_shape,dtype=np.uint8)
        
    #now make truth graph y (i.e. both hits are sim-matched)    
    y = (np.isin(pairs_sel,sim_indices).astype(np.int8).sum(axis=-1) == 2)
    
    return R_i,R_o,y




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

    print(y.shape)
    print(y)
    # sys.exit()

    return Ri, Ro, y



verbose = True
def make_graph_knn_old(coords, layers, sim_indices, k):

    if verbose: print(coords.shape, layers.shape, sim_indices.shape)

    nbrs = NearestNeighbors(algorithm='kd_tree').fit(coords)
    if verbose: print('nbrs', nbrs)

    nbrs_sm = nbrs.kneighbors_graph(coords, k)

    # Make undirected
    nbrs_sm = symmetrize_csr(nbrs_sm)

    nbrs_sm.setdiag(0) #remove self-loop edges
    nbrs_sm.eliminate_zeros() 

    # nbrs_sm = nbrs_sm + nbrs_sm.T
    # if verbose: print('nbrs_sm.shape', nbrs_sm.shape)

    pairs_sel = np.array(nbrs_sm.nonzero()).T
    if verbose: print('pairs_sel.shape', pairs_sel.shape)

    first,second = pairs_sel[:,0],pairs_sel[:,1]  
    if verbose: print('first.shape', first.shape)
    if verbose: print('second.shape', second.shape)

    #selected index pair list that we label as connected
    data_sel = np.ones(pairs_sel.shape[0])
    if verbose: print('data_sel.shape', data_sel.shape)
    
    #prepare the input and output matrices (already need to store sparse)
    r_shape = (coords.shape[0],pairs_sel.shape[0])
    if verbose: print('r_shape', r_shape)

    eye_edges = np.arange(pairs_sel.shape[0])
    if verbose: print('eye_edges', eye_edges)
    
    R_i = csr_matrix((data_sel,(pairs_sel[:,1],eye_edges)),r_shape,dtype=np.uint8)
    R_o = csr_matrix((data_sel,(pairs_sel[:,0],eye_edges)),r_shape,dtype=np.uint8)
    if verbose: print('R_i.shape', R_i.shape)
    if verbose: print('R_o.shape', R_o.shape)
        
    #now make truth graph y (i.e. both hits are sim-matched)    
    y = (np.isin(pairs_sel,sim_indices).astype(np.int8).sum(axis=-1) == 2)
    if verbose: print('y.shape', y.shape)

    return R_i,R_o,y    
        


def make_graph_xy(arrays, valid_sim_indices, ievt, mask, layered_norm, algo, preprocessing_args):
   
    x = arrays[b'rechit_x'][ievt][mask]
    y = arrays[b'rechit_y'][ievt][mask]
    z = arrays[b'rechit_z'][ievt][mask]
    layer = arrays[b'rechit_layer'][ievt][mask]
    time = arrays[b'rechit_time'][ievt][mask]
    energy = arrays[b'rechit_energy'][ievt][mask]    
    feats = np.stack((x,y,layer,time,energy)).T


    all_sim_hits = np.unique(valid_sim_indices[ievt].flatten())
    sim_hits_mask = np.zeros(arrays[b'rechit_z'][ievt].size, dtype=np.bool)
    sim_hits_mask[all_sim_hits] = True
    simmatched = np.where(sim_hits_mask[mask])[0]
    
    Ri, Ro, y_label = algo(np.stack((x,y,layer)).T, layer, simmatched, **preprocessing_args)
    
    return Graph(feats, Ri, Ro, y_label, simmatched)

def make_graph_etaphi(arrays, valid_sim_indices, ievt, mask, layered_norm, algo, preprocessing_args):
   
    x = arrays[b'rechit_x'][ievt][mask]
    y = arrays[b'rechit_y'][ievt][mask]
    z = arrays[b'rechit_z'][ievt][mask]
    layer = arrays[b'rechit_layer'][ievt][mask]
    time = arrays[b'rechit_time'][ievt][mask]
    energy = arrays[b'rechit_energy'][ievt][mask]    
    feats = np.stack((x,y,layer,time,energy)).T

    eta = arrays[b'rechit_eta'][ievt][mask]
    phi = arrays[b'rechit_phi'][ievt][mask]
    layer_normed = layer / layered_norm
    
    all_sim_hits = np.unique(valid_sim_indices[ievt].flatten())
    sim_hits_mask = np.zeros(arrays[b'rechit_z'][ievt].size, dtype=np.bool)
    sim_hits_mask[all_sim_hits] = True
    simmatched = np.where(sim_hits_mask[mask])[0]
    
    #Ri, Ro, y_label = make_graph_kdtree(np.stack((eta, phi, layer_normed)).T, layer, simmatched, r=r)
    Ri, Ro, y_label = algo(np.stack((eta, phi, layer_normed)).T, layer, simmatched, **preprocessing_args)
    
    return Graph(feats, Ri, Ro, y_label, simmatched)