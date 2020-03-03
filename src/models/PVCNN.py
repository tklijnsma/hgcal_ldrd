import os, sys, logging
import torch
import torch.nn as nn

if 'glue' in logging.Logger.manager.loggerDict:
    logger = logging.getLogger('glue')
else:
    # Make new logger
    logger = logging.getLogger('pvcnnlogger')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt = (
            '\033[33m' # Make yellow color in terminal
            '%(levelname)8s:%(asctime)s:%(module)s:%(lineno)s'
            '\033[0m'
            ' %(message)s'
            ),
        datefmt='%Y-%m-%d %H:%M:%S'
        ))
    logger.addHandler(handler)

try:
    from modules.pvconv import PVConv
    # from models.utils import create_mlp_components, create_pointnet_components
    from modules import SharedMLP, PVConv
    continue_importing = True
except ImportError:
    logger.error('Could not import pvcnn')
    continue_importing = False

PVConvForHGCAL = None

if continue_importing:
    # class PVConvForHGCAL(PVConv):

    #     def split_hgcal_features_into_positions_and_features(self, hgcal_features):
    #         # hgcal_features: (n_batch, n_features)
    #         logger.debug('hgcal_features.size() = %s', hgcal_features.size())
    #         positions, features = torch.split(hgcal_features, 3, dim=1)
    #         logger.debug('positions.size() = %s', positions.size())
    #         logger.debug('features.size() = %s', features.size())
    #         sys.exit()

    #     def merge_positions_and_features_into_hgcal_features(self, positions, hgcal_features):
    #         raise NotImplementedError

    #     def forward(self, data):
    #         all_hgcal_features = data.x
    #         # Split features into x, y, z and E, t here
    #         positions, features = self.split_hgcal_features_into_positions_and_features(
    #             all_hgcal_features
    #             )
    #         # Do the update according to PVConv
    #         features, positions = super(PVConvForHGCAL, self).forward(features, positions)
    #         # Remerge back into one Graph data structure
    #         all_hgcal_features_updated = self.merge_positions_and_features_into_hgcal_features(
    #             positions, features
    #             )
    #         return all_hgcal_features_updated


    class PVConvForHGCAL(nn.Module):

        def __init__(self, num_classes, in_channels):
            super(PVConvForHGCAL, self).__init__()

            self.point_features = nn.ModuleList([
                # Block 1
                PVConv(in_channels = in_channels, out_channels = 64, resolution = 32, kernel_size = 3),
                # Block 2
                PVConv(in_channels = 64, out_channels = 64,  resolution = 16, kernel_size = 3),
                PVConv(in_channels = 64, out_channels = 64,  resolution = 16, kernel_size = 3),
                # Block 3
                PVConv(in_channels = 64, out_channels = 128, resolution = 16, kernel_size = 3),
                # Block 4
                SharedMLP(in_channels = 128, out_channels = 1024)
                ])

            self.cloud_features = nn.Sequential(
                nn.Sequential(
                    nn.Linear(1024, 256),
                    # nn.BatchNorm1d(256),
                    nn.ReLU(True)
                    ),
                nn.Sequential(
                    nn.Linear(256, 128),
                    # nn.BatchNorm1d(128),
                    nn.ReLU(True)
                    ),
                )

            in_channels = (
                64 + 64 + 64 + 128 + 1024 # All out_channels of the point_features
                + 128 # Last out_channels of the cloud_features
                )

            self.classifier = nn.Sequential(
                SharedMLP(in_channels = in_channels, out_channels = 512),
                nn.Dropout(0.3),
                SharedMLP(in_channels = 512, out_channels = 256),
                nn.Dropout(0.3),
                nn.Conv1d(256, num_classes, 1)
                )

        def forward(self, data):
            logger.debug('data = %s', data)
            logger.debug('data.x = %s', data.x)
            coords, energy_and_time = torch.split(data.x, 3, dim=1)
            logger.debug('coords.size() = %s', coords.size())
            logger.debug('energy_and_time.size() = %s', energy_and_time.size())
            inputs = data.x

            coords = torch.t(coords)
            inputs = torch.t(inputs)
            logger.debug('coords.size() = %s', coords.size())

            coords = torch.unsqueeze(coords, 0)
            inputs = torch.unsqueeze(inputs, 0)
            logger.debug('coords.size() = %s', coords.size())

            # Run the point features
            intermediate_features = []
            for i_layer, layer in enumerate(self.point_features):
                logger.debug('Running layer %s', i_layer)
                # Run a point_features layer
                updated_features, the_same_coordinates = layer((inputs, coords))
                # Store the intermediate output
                intermediate_features.append(updated_features)
                # Overwrite inputs for the next layer
                inputs = updated_features

            # Run cloud features
            logger.debug('Running cloud features')
            logger.debug('inputs.size() = %s', inputs.size())
            logger.debug('inputs.max(dim=-1, keepdim=False).values.size() = %s', inputs.max(dim=-1, keepdim=False).values.size())
            inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)

            # Add features from cloud_features also the stored intermediate output
            logger.debug('Adding cloud features to list of all intermediate features')
            intermediate_features.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))

            # Run classifier on all the intermediate features
            logger.debug('Running classifier')
            logger.debug('Stored intermediate features:')
            for i, item in enumerate(intermediate_features):
                logger.debug('#%s: size = %s', i, item.size())

            logger.debug('torch.cat(intermediate_features, dim=1).size() = %s', torch.cat(intermediate_features, dim=1).size())
            output = self.classifier(torch.cat(intermediate_features, dim=1))
            logger.debug('output.size() = %s', output.size())
            # Remove the (1, ...), and transpose to (n_hits, n_categories)
            output = torch.t(output.squeeze(0))
            logger.debug('output.size() = %s', output.size())
            return output



# ________________________________________________________________________________
# DIRECT COPY FROM PVCNN STUFF

if False: # Don't run this

    import functools

    class PVCNN_s3dis(nn.Module):
        blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

        def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1):
            super().__init__()
            self.in_channels = extra_feature_channels + 3

            layers, channels_point, concat_channels_point = create_pointnet_components(
                blocks=self.blocks, in_channels=self.in_channels, with_se=False,
                width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
            )
            self.point_features = nn.ModuleList(layers)

            layers, channels_cloud = create_mlp_components(
                in_channels=channels_point, out_channels=[256, 128],
                classifier=False, dim=1, width_multiplier=width_multiplier)
            self.cloud_features = nn.Sequential(*layers)

            layers, _ = create_mlp_components(
                in_channels=(concat_channels_point + channels_cloud),
                out_channels=[512, 0.3, 256, 0.3, num_classes],
                classifier=True, dim=2, width_multiplier=width_multiplier
            )
            self.classifier = nn.Sequential(*layers)

        def forward(self, inputs):
            if isinstance(inputs, dict):
                inputs = inputs['features']

            coords = inputs[:, :3, :]
            out_features_list = []
            for i in range(len(self.point_features)):
                inputs, _ = self.point_features[i]((inputs, coords))
                out_features_list.append(inputs)
            # inputs: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
            inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
            out_features_list.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
            return self.classifier(torch.cat(out_features_list, dim=1))



    def _linear_bn_relu(in_channels, out_channels):
        return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


    def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
        r = width_multiplier

        if dim == 1:
            block = _linear_bn_relu
        else:
            block = SharedMLP
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
            return nn.Sequential(), in_channels, in_channels

        layers = []
        for oc in out_channels[:-1]:
            if oc < 1:
                layers.append(nn.Dropout(oc))
            else:
                oc = int(r * oc)
                layers.append(block(in_channels, oc))
                in_channels = oc
        if dim == 1:
            if classifier:
                layers.append(nn.Linear(in_channels, out_channels[-1]))
            else:
                layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
        else:
            if classifier:
                layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
            else:
                layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
        return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


    def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
        r, vr = width_multiplier, voxel_resolution_multiplier

        layers, concat_channels = [], 0
        for out_channels, num_blocks, voxel_resolution in blocks:
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                          with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                layers.append(block(in_channels, out_channels))
                in_channels = out_channels
                concat_channels += out_channels
        return layers, in_channels, concat_channels

    logger.info('Printing their model')
    model = PVCNN_s3dis(num_classes=4)
    logger.info(
        'Model: \n%s\nParameters: %i',
        model, sum(p.numel() for p in model.parameters())
        )
    logger.info('------------------------------------')

