import os, os.path as osp, math, numpy as np, tqdm, logging, pprint, shutil, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (
    NNConv, graclus, max_pool, max_pool_x,
    global_mean_pool
    )
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ___________________________________________________________
def setup_logger(name='hgcal', fmt=None):
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

# ___________________________________________________________
CACHED_FUNCTIONS = []
def cache_return_value(func):
    """
    Decorator that only calls a function once, and
    subsequent calls just return the cached return value
    """
    global CACHED_FUNCTIONS
    def wrapper(*args, **kwargs):
        if not getattr(wrapper, 'is_called', False) or not hasattr(wrapper, 'cached_return_value'):
            wrapper.is_called = True
            wrapper.cached_return_value = func(*args, **kwargs)
            CACHED_FUNCTIONS.append(wrapper)
        else:
            logger.debug(
                'Returning cached value for %s: %s',
                func.__name__, wrapper.cached_return_value
                )
        return wrapper.cached_return_value
    return wrapper

def clear_cache():
    global CACHED_FUNCTIONS
    for func in CACHED_FUNCTIONS:
        func.is_called = False
    CACHED_FUNCTIONS = []

# ___________________________________________________________
from datasets.hitgraphs import HitGraphDataset
from torch_geometric.data import Data
class MNISTDataset(HitGraphDataset):

    def __init__(self, *args, **kwargs):
        super(MNISTDataset, self).__init__(*args, **kwargs)
        self.is_loaded = False
        self.image_size = 28
        self.x_oneimage = np.tile(np.arange(self.image_size), self.image_size)
        self.y_oneimage = np.repeat(np.arange(self.image_size-1,-1,-1), self.image_size)
        self.graphs = []

    def process(self):
        pass

    def load(self):
        if self.is_loaded: return
        for key in [ 'train', 'test' ]:
            if key == 'test': continue # Skip it for now
            images_file = [f for f in self.raw_paths if (key in f and 'image' in f)][0]
            labels_file = [f for f in self.raw_paths if (key in f and 'label' in f)][0]
            logger.debug('Processing %s images file %s', key, images_file)
            logger.debug('Processing %s labels file %s', key, labels_file)
            with np.load(images_file) as npdata:
                images = npdata['img']
                n_images = images.shape[0]
                values = np.reshape(images, (n_images, self.image_size**2))
                x = np.repeat(self.x_oneimage.reshape((1,self.x_oneimage.shape[0])), n_images, axis=0)
                y = np.repeat(self.y_oneimage.reshape((1,self.y_oneimage.shape[0])), n_images, axis=0)
                assert values.shape == (n_images, self.image_size**2)
                assert x.shape == (n_images, self.image_size**2)
                assert y.shape == (n_images, self.image_size**2)
                features = np.stack((x, y, values), axis=-1)
                assert features.shape == (n_images, self.image_size**2, 3)
            with np.load(labels_file) as npdata:
                labels = npdata['label'].reshape((n_images, 1))
                assert labels.shape == (n_images, 1)
            for i_img in range(n_images):
                # Put all graphs as Data objects in memory
                self.graphs.append(Data(
                    x = torch.from_numpy(features[i_img]).type(torch.FloatTensor),
                    y = torch.from_numpy(labels[i_img]).type(torch.LongTensor)
                    ))
        self.is_loaded = True

    def __len__(self):
        self.load()
        return len(self.graphs)

    def get(self, i):
        self.load()
        return self.graphs[i]



class LindseysTrainingScript(object):
    """docstring for LindseysTrainingScript"""
    def __init__(self, debug=True):
        super(LindseysTrainingScript, self).__init__()
        self.debug = debug

        self.directed = False
        # self.train_batch_size = 1
        # self.valid_batch_size = 1
        if self.debug:
            self.train_batch_size = 2
            self.valid_batch_size = 2
        else:
            self.train_batch_size = 32
            self.valid_batch_size = 32

        self.categorized = True
        self.forcecats = True
        self.cats = 10
        self.model_name = 'DynamicReductionNetwork'
        # self.loss = 'nll_loss'
        self.loss = 'cross_entropy'

        self.optimizer = 'AdamW'
        self.hidden_dim = 64
        self.n_iters = 6
        self.lr = 1e-3
        self.output_dir = osp.join(THISDIR, 'output')

        if self.debug:
            self.n_epochs = 3
            self.dataset_path = osp.join(THISDIR, 'mnistdata')
            logger.setLevel(logging.DEBUG)
        else:
            self.n_epochs = 30
            self.dataset_path = osp.join(THISDIR, 'mnistdata')
            logger.setLevel(logging.INFO)

        self.load_checkpoint = None


    @cache_return_value
    def get_full_dataset(self):
        if not osp.isdir(self.dataset_path):
            raise OSError('{0} is not a valid path'.format(self.dataset_path))

        if self.debug:
            processed_path = osp.join(self.dataset_path, 'processed')
            if osp.isdir(processed_path):
                logger.warning('Test sample: Removing %s to force reprocessing', processed_path)
                shutil.rmtree(processed_path)

        logger.info('Using dataset_path %s', self.dataset_path)
        full_dataset = MNISTDataset(
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
        return full_dataset, train_dataset, valid_dataset


    def get_num_features(self, full_dataset):
        num_features = full_dataset.num_features
        logger.debug('num_features = %s', num_features)
        return num_features


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

    @cache_return_value
    def get_trainer(self):
        full_dataset, train_dataset, valid_dataset = self.get_full_dataset()
        num_features = self.get_num_features(full_dataset)
        num_classes = self.get_num_classes(full_dataset)
        trainer = self._get_trainer(num_classes, num_features)
        return trainer

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
        
        if 'pvcnn' in self.model_name.lower():
            model_args = {
                'in_channels': num_features,  # x, y, z, E, t
                'num_classes': num_classes,
                }
        elif 'dynamicreductionnetwork' in self.model_name.lower():
            model_args = {
                'input_dim'     : 3,
                'hidden_dim'    : self.hidden_dim,
                'output_dim'    : num_classes,
                'norm'          : torch.tensor([1./28., 1./28., 1.]),
                'k'             : 8
                }
        else:
            model_args = {
                'input_dim'     : num_features,
                'hidden_dim'    : self.hidden_dim,
                'n_iters'       : self.n_iters,
                'output_dim'    : num_classes,
                # 'norm'          : torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.]),
                'norm'          : torch.tensor([1./10., 1./10.]),
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
    debug = False
    # debug = True
    script = LindseysTrainingScript(debug=debug)
    # script.load_checkpoint = 'Mar20_epoch0-14/output/checkpoints/model_checkpoint_EdgeNetWithCategories_256834_3e794862ba_klijnsma_014.pth.tar'
    script.train()


if __name__ == '__main__':
    main()