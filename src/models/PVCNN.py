import os, sys, logging

logger = logging.getLogger('pvcnnlogger')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt = (
        '\033[33m'
        '%(levelname)8s:%(asctime)s:%(module)s:'
        '\033[0m'
        ' %(message)s'
        ),
    datefmt='%Y-%m-%d %H:%M:%S'
    ))
logger.addHandler(handler)


if 'PVCNNPATH' in os.environ:
    pvcnnpath = os.environ[PVCNNPATH]
    logger.info('Attempting to import pvcnn from %s', pvcnnpath)
    if not pvcnnpath in sys.path: sys.path.append(pvcnnpath)

    try:
        from modules.pvconv import PVConv
        continue_importing = True
    except ImportError:
        logger.error('Could not import pvcnn')
        continue_importing = False

    if continue_importing:

        class PVConvForHGCAL(PVConv):

            def split_hgcal_features_into_positions_and_features(self, hgcal_features):
                raise NotImplementedError

            def merge_positions_and_features_into_hgcal_features(self, positions, hgcal_features):
                raise NotImplementedError

            def forward(self, data):
                all_hgcal_features = data.X
                # Split features into x, y, z and E, t here
                positions, features = self.split_hgcal_features_into_positions_and_features(
                    all_hgcal_features
                    )
                # Do the update according to PVConv
                features, positions = super(PVConvForHGCAL, self).forward(features, positions)
                # Remerge back into one Graph data structure
                all_hgcal_features_updated = self.merge_positions_and_features_into_hgcal_features(
                    positions, features
                    )
                return all_hgcal_features_updated
