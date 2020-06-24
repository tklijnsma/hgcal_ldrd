import uproot
import awkward
import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm
import os, os.path as osp, sys, glob, multiprocessing as mp
sys.path.append('/home/klijnsma/basehgcal/particle-number-truth/hgcal_ldrd/src')

from scipy.sparse import coo_matrix # to encode the cluster mappings
from sklearn.neighbors import NearestNeighbors
from datasets.graph import Graph
from datasets.graph import graph_to_sparse, save_graph

class PreprocessRoot(object):
    def __init__(self, rootfile):
        self.rootfile = osp.realpath(rootfile)
        self.outdir = osp.join(
            osp.dirname(osp.dirname(self.rootfile)),
            'rawpointclouds'
            )
        self.progress_bar = True
        self.rootfile_number = int(self.rootfile.split('_')[-1].replace('.root',''))-1

    def get_category(self, pid):
        cats = np.zeros_like(pid) # 1 are hadrons
        cats[(pid == 22) | (np.abs(pid) == 11) | (pid == 111)] = 1 # 2 are EM showers
        cats[np.abs(pid) == 13] = 2 #3 are MIPs
        return (cats+1) # category zero are the noise hits

    def get_features(self, ievt, mask):
        x = self.rechit_x[ievt][mask]
        y = self.rechit_y[ievt][mask]
        layer = self.rechit_layer[ievt][mask]
        time = self.rechit_time[ievt][mask]
        energy = self.rechit_energy[ievt][mask]    
        return np.stack((x,y,layer,time,energy)).T.astype(np.float32)

    def dump(self):
        print('Dumping', self.rootfile)
        test = uproot.open(self.rootfile)['ana']['hgc']

        sim_indices = awkward.fromiter(test['simcluster_hits_indices'].array())
        sim_indices = sim_indices[sim_indices > -1].compact()
        sim_energy = test['simcluster_energy'].array()
        sim_pid = test['simcluster_pid'].array()

        self.rechit_layer = test['rechit_layer'].array()
        self.rechit_time = test['rechit_time'].array()
        self.rechit_energy = test['rechit_energy'].array()
        self.rechit_x = test['rechit_x'].array()
        self.rechit_y = test['rechit_y'].array()
        self.rechit_z = test['rechit_z'].array()

        if not osp.isdir(self.outdir): os.makedirs(self.outdir)
            
        iterator = range(self.rechit_z.size)
        if self.progress_bar: iterator = tqdm(iterator, desc='events processed')

        for i in iterator:
            cluster_cats = self.get_category(sim_pid[i])
                    
            sim_indices_cpt = awkward.fromiter(sim_indices[i])
            if isinstance(sim_indices_cpt, np.ndarray):
                if sim_indices_cpt.size == 0: #skip events that are all noise, they're meaningless anyway
                    continue
                else:
                    sim_indices_cpt = awkward.JaggedArray.fromcounts([sim_indices_cpt.size],sim_indices_cpt)
            hits_in_clus = sim_indices_cpt.flatten()
            hit_to_clus = sim_indices_cpt.parents
            
            # 0 = invalid edge, 1 = hadronic edge, 2 = EM edge, 3 = MIP edge 
            cats_per_hit = cluster_cats[hit_to_clus]

            pos_mask = (self.rechit_z[i] > 0)
            neg_mask = ~pos_mask
            
            pos_feats = self.get_features(i, pos_mask)
            neg_feats = self.get_features(i, neg_mask)

            # Construct y per point: [ category, cluster_number ]
            y = np.zeros((2, self.rechit_z[i].shape[0]))
            y[0][hits_in_clus] = cats_per_hit
            y[1][hits_in_clus] = hit_to_clus + 1 # Reserve cluster 0 to be all the noise hits
            y = y.T.astype(np.int)
            
            assert y.shape == ( self.rechit_z[i].shape[0], 2 )

            y_pos = y[pos_mask]
            y_neg = y[neg_mask]
            
            assert y_pos.shape[0] == pos_feats.shape[0]
            assert y_neg.shape[0] == neg_feats.shape[0]
            
            np.savez(
                osp.join(self.outdir, '{0:05d}_pos.npz'.format(i + 1000*self.rootfile_number)),
                x = pos_feats,
                y = y_pos
                )
            np.savez(
                osp.join(self.outdir, '{0:05d}_neg.npz'.format(i + 1000*self.rootfile_number)),
                x = neg_feats,
                y = y_neg
                )


def dump_rootfile(rootfile):
    PreprocessRoot(rootfile).dump()

def dump_rootfile_mp(rootfile):
    preproc = PreprocessRoot(rootfile)
    preproc.progress_bar = False
    preproc.dump()

def main():
    processes = []
    for rootfile in glob.glob('single-tau-ntup/*.root'):
        # dump_rootfile(rootfile)
        p = mp.Process(target=dump_rootfile_mp, args=(rootfile,))
        p.start()
        processes.append(p)
    [ p.join() for p in processes ]

if __name__ == '__main__':
    main()