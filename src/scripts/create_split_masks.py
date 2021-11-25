from pathlib import Path

import rasterio
import numpy as np

STRIP_WIDTH = 180
STRIPS_TRAIN = 3
STRIPS_VAL = 1
STRIPS_TEST = 1

DATA_PATH = Path('/home/pf/pfstaff/data/NINA_forest_structure/preprocessed')
OUT_PATH = Path('/home/pf/pfstaff/data/NINA_forest_structure/split_masks')

OUT_PATH.mkdir(exist_ok=True)
count_train, count_val, count_test = 0, 0, 0

for path in DATA_PATH.glob('*.tif'):
    if path.name.startswith('.'):
        print(f'Skipping {path.name}')
        continue

    print(f'Processing {path.name}')

    src = rasterio.open(path)
    shape = src.shape
    valid_mask = src.read_masks(1)

    split_mask = np.zeros((1, shape[0], shape[1]))
    n_strips = STRIPS_TRAIN + STRIPS_VAL + STRIPS_TEST

    for i in range(shape[0]):
        i_mod = (i % (STRIP_WIDTH * n_strips)) // STRIP_WIDTH

        if i_mod < STRIPS_TRAIN:
            value = 0.
        elif i_mod < STRIPS_TRAIN + STRIPS_VAL:
            value = 1.
        else:
            value = 2.

        split_mask[0, i] = value

    # count values
    for i in range(shape[0]):
        for j in range(shape[1]):
            if valid_mask[i, j]:
                value = split_mask[0, i, j]
                if value == 0:
                    count_train += 1
                elif value == 1:
                    count_val += 1
                elif value == 2:
                    count_test += 1
                else:
                    raise ValueError(f'Encountered invalid split value: {value}')

    # save
    with rasterio.Env():
        profile = src.profile

        profile.update(
            driver='GTiff',
            count=1,
            compress='deflate',
            dtype=rasterio.uint8,
            nodata=None)

        with rasterio.open(OUT_PATH / path.name, 'w', **profile) as f:
            f.write(split_mask.astype('uint8'))

_sum = count_train + count_val + count_test
print(f'Total valid data points: {_sum}')
print(f'Distribution: {np.array([count_train, count_val, count_test]) / _sum}')
