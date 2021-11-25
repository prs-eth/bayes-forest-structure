import sys
from pathlib import Path
from itertools import chain
import pickle
import yaml

import numpy as np
from tqdm import trange

sys.path.append('repo/src')
from utils import split_list, RunningStats

SEPARATOR = 65535  # max uint16 value
PKL_PATH = Path(sys.argv[1])

# recover patch size from data_config.yaml
with (PKL_PATH / 'data_config.yaml').open('r') as fh:
    patch_size = yaml.safe_load(fh)['patch_size']

s2_stats = RunningStats((12,))
s1_stats = RunningStats((2,))
labels_stats = RunningStats((5,))

for pkl_file in PKL_PATH.glob('*.pkl'):
    print(f'Processing {pkl_file.name}...')

    with pkl_file.open('rb') as fh:
        data = pickle.load(fh)

    locations = data['train'][0]
    loc_to_images_map = data['train'][1]
    offsets = data['train'][2]
    images = data['images']
    labels = data['labels']

    for index in trange(len(locations)):
        i, j = locations[index]
        patch_half = patch_size // 2
        i_slice = slice(i - patch_half, i + patch_half + 1)
        j_slice = slice(j - patch_half, j + patch_half + 1)

        # extract the part from loc_to_images_map that contains the valid s2 and s1 indices for this location
        upper = None if index == len(locations) - 1 else offsets[index + 1]
        _map = loc_to_images_map[offsets[index]:upper]
        assert _map[0] != SEPARATOR and _map[-1] == SEPARATOR, f"{i}, {j}"

        _map = np.array(split_list(_map.tolist(), SEPARATOR), dtype='object')
        assert len(_map) % 3 == 0

        for s2_indices, s1_a_indices, s1_d_indices in np.array_split(_map, len(_map) // 3):
            assert len(s2_indices) == 1
            s2_index = s2_indices[0]

            s2_stats.add(images[s2_index][:12, i_slice, j_slice].reshape(-1, patch_size**2).transpose())
            for s1_index in chain(s1_a_indices, s1_d_indices):
                s1_stats.add(images[s1_index][:, i_slice, j_slice].reshape(-1, patch_size**2).transpose())

        labels_patch = labels[:, i_slice, j_slice]
        labels_stats.add(labels_patch[:, ~np.isnan(labels_patch).any(0)].transpose())

with (PKL_PATH / 'stats.yaml').open('r') as fh:
    stats = yaml.safe_load(fh)
    stats.update({
        's2_mean': s2_stats.mean.tolist(), 's2_std': s2_stats.std.tolist(),
        's1_mean': s1_stats.mean.tolist(), 's1_std': s1_stats.std.tolist(),
        'labels_mean': labels_stats.mean.tolist(), 'labels_std': labels_stats.std.tolist()
    })
with (PKL_PATH / 'stats.yaml').open('w') as fh:
    yaml.dump(stats, fh)
