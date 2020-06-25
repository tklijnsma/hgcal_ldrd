import sys, shutil, time
import os, os.path as osp, math, numpy as np, tqdm, logging, pprint, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (
    NNConv, graclus, max_pool, max_pool_x,
    global_mean_pool
    )
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.neighbors import NearestNeighbors

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ___________________________________________________________
def setup_logger(name='hcal', fmt=None):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.warning('Logger %s is already defined', name)
    else:
        if fmt is None:
            fmt = logging.Formatter(
                fmt = '\033[33m%(levelname)8s:%(asctime)s:%(module)s:%(lineno)s\033[0m %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
                )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

# ___________________________________________________________
THISDIR = osp.abspath(osp.dirname(__file__))
sys.path.append(osp.join(THISDIR, '../src'))
from datasets.hitgraphs import HitGraphDataset


class HCALDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(
            self,
            root,
            directed = True,
            categorical = False,
            transform = None,
            pre_transform = None
            ):
        self._directed = directed
        self._categorical = categorical
        super(HCALDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(self.raw_dir+'/*.npz')
        return [osp.basename(f) for f in self.input_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx, raw_path in enumerate(tqdm(self.raw_paths)):
            npz = np.load(raw_path)
            x = npz['x']
            y = npz['y']
            z = npz['z']
            energy = npz['energy']
            time = npz['time']
            hitlabel = npz['hit']
            n_hits = x.shape[0]

            features = np.stack((x, y, z, energy, time)).T
            coordinates = np.stack((x, y, z)).T

            k = 16

            nbrs = NearestNeighbors(algorithm='kd_tree').fit(coordinates)
            nbrs_sm = nbrs.kneighbors_graph(coordinates, k)
            nbrs_sm.setdiag(0) #remove self-loop edges
            nbrs_sm.eliminate_zeros() 
            nbrs_sm = nbrs_sm + nbrs_sm.T # diagonalize it (entries will be 1 or 2)
            node_index_in, node_index_out = nbrs_sm.nonzero()

            # Create the edge_index array in the format the model expects
            n_edges = node_index_in.shape[0]
            edge_index = np.stack((node_index_in, node_index_out))
            assert edge_index.shape == (2, n_edges)

            # Compute the label for the edge: Only True if both the in- and out-hits are signal (i.e. 1)
            edgelabel = hitlabel[edge_index.T].all(axis=1)
            assert edgelabel.shape == (n_edges,)

            outdata = Data(
                x = torch.from_numpy(features).type(torch.FloatTensor),
                edge_index = torch.from_numpy(edge_index).type(torch.LongTensor),
                y = torch.from_numpy(edgelabel).type(torch.LongTensor)
                )
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))


class TrainingScript(object):
    def __init__(self, debug=True):
        super(TrainingScript, self).__init__()
        self.debug = debug

        self.directed = False
        self.train_batch_size = 1
        self.valid_batch_size = 1

        # Do a 'categorized' training with two cats
        # This is basically a binary problem, but I don't know if you want to extend this to
        # more categories at some point?
        self.categorized = True
        self.forcecats = True
        self.cats = 2
        self.model_name = 'EdgeNetWithCategories'
        self.loss = 'nll_loss'

        self.optimizer = 'AdamW'
        self.hidden_dim = 32
        self.n_iters = 5
        self.lr = 1e-3
        self.output_dir = osp.join(THISDIR, '../output')

        data_dir = osp.join(THISDIR, '../../data')
        if self.debug:
            self.n_epochs = 3
            self.dataset_path = osp.join(data_dir, 'jeffhcal-debug')
            logger.setLevel(logging.DEBUG)
        else:
            self.n_epochs = 45
            self.dataset_path = osp.join(data_dir, 'jeffhcal')
            logger.setLevel(logging.INFO)

        self.load_checkpoint = None
        self._has_full_dataset = False
        self._has_trainer = False

    def get_full_dataset(self):
        if self._has_full_dataset: return self.full_dataset, self.train_dataset, self.valid_dataset
        if not osp.isdir(self.dataset_path):
            raise OSError('{0} is not a valid path'.format(self.dataset_path))

        if self.debug and 'debug' in self.dataset_path:
            processed_path = osp.join(self.dataset_path, 'processed')
            logger.warning('Debug sample: Removing %s to force reprocessing', processed_path)
            shutil.rmtree(processed_path)

        logger.info('Using dataset_path %s', self.dataset_path)
        full_dataset = HCALDataset(
            self.dataset_path,
            directed = self.directed,
            categorical = self.categorized
            )
        fulllen = len(full_dataset)
        tv_frac = 0.20
        tv_num = math.ceil(fulllen*tv_frac)
        splits = np.cumsum([fulllen-tv_num,0,tv_num])

        if self.debug:
            logger.debug('Running on 7 training events, 3 validation events for debugging')
            splits = [ 0, 7, 10 ]

        logger.info('%s, %s', fulllen, splits)
        train_dataset = torch.utils.data.Subset(
            full_dataset,
            list(range(0, splits[1]))
            )
        valid_dataset = torch.utils.data.Subset(
            full_dataset,
            list(range(splits[1], splits[2]))
            )

        self.full_dataset = full_dataset
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self._has_full_dataset = True
        return self.full_dataset, self.train_dataset, self.valid_dataset

    def get_num_features(self, full_dataset):
        return full_dataset.num_features

    def get_num_classes(self, full_dataset):
        if self.categorized:
            if not self.forcecats:
                num_classes = \
                    int(full_dataset[0].y.max().item()) + 1 \
                    if full_dataset[0].y.dim() == 1 else full_dataset[0].y.size(1)
            else:
                num_classes = self.cats
        logger.debug('num_classes = %s', num_classes)
        return num_classes

    def get_trainer(self):
        if not self._has_trainer:
            full_dataset, train_dataset, valid_dataset = self.get_full_dataset()
            num_features = self.get_num_features(full_dataset)
            num_classes = self.get_num_classes(full_dataset)
            self.trainer = self._get_trainer(num_classes, num_features)
            self._has_trainer = True
        return self.trainer

    def _get_trainer(self, num_classes, num_features):
        from training.gnn import GNNTrainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('using device %s', device)

        trainer = GNNTrainer(
            category_weights = np.ones(num_classes), 
            output_dir = self.output_dir,
            device = device
            )
        trainer.logger.setLevel(logging.DEBUG)
        trainer.logger.addHandler(logger.handlers[0]) # Just give same handler as other log messages

        #example lr scheduling definition
        def lr_scaling(optimizer):
            from torch.optim.lr_scheduler import ReduceLROnPlateau        
            return ReduceLROnPlateau(
                optimizer, mode='min', verbose=True,
                min_lr = 5e-7, factor = 0.2, 
                threshold = 0.05, patience = 5
                )
        
        model_args = {
            'input_dim'     : num_features,
            'hidden_dim'    : self.hidden_dim,
            'n_iters'       : self.n_iters,
            'output_dim'    : num_classes
            }

        trainer.build_model(
            name          = self.model_name,
            loss_func     = self.loss,
            optimizer     = self.optimizer,
            learning_rate = self.lr,
            lr_scaling    = lr_scaling,
            **model_args
            )
        trainer.print_model_summary()

        if self.load_checkpoint:
            logger.warning('Loading weights from previous checkpoint: %s', self.load_checkpoint)
            trainer.model.load_state_dict(torch.load(self.load_checkpoint)['model'])

        return trainer

    def train(self):
        full_dataset, train_dataset, valid_dataset = self.get_full_dataset()
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False)
        trainer = self.get_trainer()
        train_summary = trainer.train(train_loader, self.n_epochs, valid_data_loader=valid_loader)
        logger.info(train_summary)

    def test(self):
        full_dataset, train_dataset, valid_dataset = self.get_full_dataset()
        valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(10)))
        valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False)
        trainer = self.get_trainer()
        summary = trainer.evaluate(valid_loader)
        logger.info('Test summary:\n%s', pprint.pformat(summary))


def main():
    script = TrainingScript(debug=True) # Debug mode runs only a few events to check for bugs
    # Here's how to load a checkpoint:
    # script.load_checkpoint = 'training-Mar06-epoch-15-29/output/checkpoints/model_checkpoint_PVConvForHGCAL_2562244_9c8b11eb88_klijnsma_014.pth.tar'
    script.train()

if __name__ == '__main__':
    main()