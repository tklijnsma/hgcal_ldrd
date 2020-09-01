import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
# from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool, global_max_pool,
                                global_add_pool)

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class DynamicReductionNetwork(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetwork, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),            
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )                
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),                                    
                                    nn.Linear(hidden_dim//2, output_dim))
        
        
    def forward(self, data):        
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)
        
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        data.x = self.edgeconv1(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow))
        data.x = self.edgeconv2(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_max_pool(x, batch)
        
        return self.output(x).squeeze(-1)

# _______________________________________________________________________________
from typing import Optional
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce

def to_undirected(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # row, col = edge_index
    row = edge_index[0,:]
    col = edge_index[1,:]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return edge_index


class DynamicReductionNetworkJittable(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetworkJittable, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),            
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )                
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr).jittable()
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr).jittable()
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),                                    
                                    nn.Linear(hidden_dim//2, output_dim))
        
        
    def forward(self, x, batch: Optional[torch.Tensor] = None):
        x = self.datanorm * x
        x = self.inputnet(x)
        

        knn = knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv1.flow)
        edge_index = to_undirected(knn)
        x = self.edgeconv1(x, edge_index)
        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        edge_attr = None
        data = max_pool(cluster, data)
        
        edge_index = to_undirected(knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv2.flow))
        x = self.edgeconv2(x, edge_index)
        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        x, batch = max_pool_x(cluster, x, batch)

        x = global_max_pool(x, batch)
        
        return self.output(x).squeeze(-1)
