import sys
import yaml
from pathlib import Path

import rasterio
import numpy as np
from latextable import Texttable, draw_latex

sys.path.append('repo/src')


PREDICTIONS_DIR = Path(sys.argv[1])
PKL_DIR = Path('data/pkl/2021-05-18_10-57-45')
GT_DIR = Path('data/preprocessed')
NUM_VARS = 5
VARIABLES = ['P95', 'MeanH', 'Dens', 'Gini', 'Cover']
METRICS = ['MAE', 'NMAE', 'RMSE', 'NRMSE', 'MBE', 'NMBE']

# regions
EAST = ['346', '9', '341', '354', '415', '418', '416', '429', '439', '560', '472', '521', '498',
        '522', '564', '764', '781', '825', '796', '805', '827', '891', '835', '920', '959', '1023', '998',
        '527', '477', '542', '471']
WEST = ['528', '537', '792', '988', '769']
NORTH = ['819', '909', '896']
ALL = EAST + WEST + NORTH

with (PKL_DIR / 'stats.yaml').open() as fh:
    # load training set statistics for data normalization
    stats = yaml.safe_load(fh)
    labels_mean = np.array(stats['labels_mean'])

maes, mses, mbes = {}, {}, {}
num_pixels = {}

for mean_file in PREDICTIONS_DIR.glob('*_mean.tif'):
    project = mean_file.stem.split('_')[0]
    if project not in ALL:
        continue

    print(f'Processing project {project}...')

    with rasterio.open(mean_file) as fh:
        mean = fh.read(fh.indexes)

    with rasterio.open(PREDICTIONS_DIR / (project + '_variance.tif')) as fh:
        variance = fh.read(fh.indexes)

    with rasterio.open(GT_DIR / (project + '.tif')) as fh:
        gt = fh.read(fh.indexes)
        gt_mask = fh.read_masks(1).astype(np.bool)

    # only evaluate pixels that are valid according to the ground truth mask, i.e. dont test on bodies of water etc.
    # also disregard pixels that are nan, indicating that no prediction was made (because of the pixels being outside
    # the test region, outside the project polygon or because no suitable images were available for prediction)
    assert mean.shape == variance.shape == gt.shape
    mask = ~np.isnan(mean).all(0)
    diff = mean[:, mask] - gt[:, mask]

    num_pixels[project] = mask.sum()
    maes[project] = np.abs(diff).mean(1)
    mses[project] = (diff**2).mean(1)
    mbes[project] = diff.mean(1)

for group, name in zip((EAST, WEST, NORTH, ALL), ('east', 'west', 'north', 'all')):
    mae_group = np.zeros((5,))
    mse_group = np.zeros((5,))
    mbe_group = np.zeros((5,))
    num_pixels_group = 0

    for project in group:
        mae_group += maes[project] * num_pixels[project]
        mse_group += mses[project] * num_pixels[project]
        mbe_group += mbes[project] * num_pixels[project]
        num_pixels_group += num_pixels[project]

    mae_group /= num_pixels_group
    mse_group /= num_pixels_group
    mbe_group /= num_pixels_group

    values = np.zeros((6, 5))
    values[0] = mae_group
    values[1] = mae_group / labels_mean
    values[2] = np.sqrt(mse_group)
    values[3] = np.sqrt(mse_group) / labels_mean
    values[4] = mbe_group
    values[5] = (mbe_group / labels_mean)

    table = Texttable()
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.set_cols_align(['c', 'c', 'c', 'c', 'c', 'c'])
    table.add_rows([['', *VARIABLES]] + [[METRICS[i]] + row for i, row in enumerate(values.tolist())])
    print(f'{name}:')
    print(table.draw())
    print()

    with open(PREDICTIONS_DIR / f'table_{name}.txt', 'w') as fh:
        fh.write(draw_latex(table))
