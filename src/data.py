import random
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List
import pickle
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import Normalize, Compose

from utils import split_list


SEPARATOR = 65535  # max uint16 value


class ForestData:
    """
    Provides DataLoader objects for the pickled NIBIO forest structure dataset.
    """

    def __init__(
            self,
            pkl_dir: str,
            projects: List[str],
            s2_image_bands: List[int],
            s1_image_bands: List[int],
            labels_bands: List[int],
            normalize_labels: bool,
            both_orbit_directions: bool,
            subsample: float,
            loader_args: dict,
    ):
        pkl_dir = Path(pkl_dir)

        locations = defaultdict(list)
        loc_to_images_map = defaultdict(list)
        offsets = defaultdict(list)
        images = []
        labels = []
        filenames = []  # for debugging

        num_projects = 0
        num_datapoints_so_far = defaultdict(int)

        for pkl_file in pkl_dir.glob('*.pkl'):
            if pkl_file.stem not in projects:
                continue
            print(f'Reading in {pkl_file.name}...')
            with pkl_file.open('rb') as fh:
                data = pickle.load(fh)

            for _set in ['train', 'val']:
                # assert that number of locations equals number of offsets
                assert data[_set][0].shape[0] == data[_set][2].shape[0]

                # some very small projects have no validation data - skip them
                if data[_set][0].shape[0] == 0:
                    continue

                # append label file index column to locations
                locations[_set].append(np.hstack(
                    [data[_set][0], np.full((len(data[_set][0]), 1), len(labels), dtype=np.uint16)]
                ))

                # add current length of loc_to_images_map to offsets and append
                offsets[_set].append(data[_set][2] + sum([len(m) for m in loc_to_images_map[_set]]))

                # concatenate locations to images maps, incrementing image indices
                data[_set][1][data[_set][1] != SEPARATOR] += len(images)
                loc_to_images_map[_set].append(data[_set][1])

                num_datapoints_so_far[_set] += len(data[_set][0])

            images.extend(data['images'])
            labels.append(data['labels'])
            filenames.append(pkl_file.name)

            num_projects += 1

        # raise exception if there is no validation data (i.e. only projects without validation data were read in)
        if num_datapoints_so_far['val'] == 0:
            raise NoValDataException()

        print(f'Read in {num_projects} project files, totaling {num_datapoints_so_far["train"]} train '
              f'and {num_datapoints_so_far["val"]} val points')

        # recover patch size from data_config.yaml
        with (pkl_dir / 'data_config.yaml').open('r') as fh:
            patch_size = yaml.safe_load(fh)['patch_size']

        # transform band indices (1-based) to array indices (0-based)
        s2_channels = (np.array(s2_image_bands) - 1).astype('int')
        s1_channels = (np.array(s1_image_bands) - 1).astype('int')
        labels_channels = (np.array(labels_bands) - 1).astype('int')
        with (pkl_dir / 'stats.yaml').open() as fh:
            stats = yaml.safe_load(fh)

        # construct standardization vectors
        self.s2_mean = np.array(stats['s2_mean'])[s2_channels].tolist()
        self.s2_std = np.array(stats['s2_std'])[s2_channels].tolist()
        self.s1_mean = np.array(stats['s1_mean'])[s1_channels].tolist()
        self.s1_std = np.array(stats['s1_std'])[s1_channels].tolist()
        if normalize_labels:
            self.labels_mean = np.array(stats['labels_mean'])[labels_channels].tolist()
            self.labels_std = np.array(stats['labels_std'])[labels_channels].tolist()
        else:
            self.labels_mean, self.labels_std = [0.] * len(labels_channels), [1.] * len(labels_channels)

        # setup data loaders
        data_args = {
            'images': images,
            'labels': labels,
            'filenames': filenames,
            'patch_size': patch_size,
            'both_orbit_directions': both_orbit_directions,
            's2_transform':
                Compose([ToTensor(), SelectChannels(s2_channels), Normalize(self.s2_mean, self.s2_std)]),
            's1_transform':
                Compose([ToTensor(), SelectChannels(s1_channels), Normalize(self.s1_mean, self.s1_std)]),
            'labels_transform':
                Compose([ToTensor(), SelectChannels(labels_channels), Normalize(self.labels_mean, self.labels_std)])
        }

        train_set, val_set = (_PatchDataset(
            np.concatenate(locations[split]),
            np.concatenate(loc_to_images_map[split]),
            np.concatenate(offsets[split]),
            **data_args
        ) for split in ('train', 'val'))

        if subsample and subsample < 1.0:
            train_sampler = RandomSampler(train_set, num_samples=int(subsample * len(train_set)))
            val_sampler = RandomSampler(val_set, num_samples=int(subsample * len(val_set)))
            shuffle = False
        else:
            train_sampler, val_sampler, shuffle = None, None, True

        self.train_loader = DataLoader(train_set, shuffle=shuffle, sampler=train_sampler, **loader_args)
        self.val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_args)


class _PatchDataset(Dataset):

    def __init__(
            self,
            locations: np.ndarray,
            loc_to_images_map: np.ndarray,
            offsets: np.ndarray,
            images: List,
            labels: List,
            filenames: List,
            patch_size: int,
            both_orbit_directions: bool,
            s2_transform: torch.nn.Module,
            s1_transform: torch.nn.Module,
            labels_transform: torch.nn.Module
    ):
        self.locations = locations
        self.loc_to_images_map = loc_to_images_map
        self.offsets = offsets
        self.images = images
        self.labels = labels
        self.filenames = filenames
        self.patch_size = patch_size
        self.both_orbit_directions = both_orbit_directions
        self.s2_transform = s2_transform
        self.s1_transform = s1_transform
        self.labels_transform = labels_transform

    def __getitem__(self, index):
        i, j, labels_index = self.locations[index].tolist()

        i_slice = slice(i - (self.patch_size // 2), i + (self.patch_size // 2) + 1)
        j_slice = slice(j - (self.patch_size // 2), j + (self.patch_size // 2) + 1)

        # extract the part from loc_to_images_map that contains the valid s2 and s1 indices for this location
        upper = None if index == len(self) - 1 else self.offsets[index + 1]
        _map = self.loc_to_images_map[self.offsets[index]:upper]
        assert _map[0] != SEPARATOR and _map[-1] == SEPARATOR, f"{i}, {j}, {self.filenames[labels_index]}"

        _map = np.array(split_list(_map.tolist(), SEPARATOR), dtype='object')
        assert len(_map) % 3 == 0

        # randomly select one S2 image & corresponding s1 ascending/descending images
        s2_indices, s1_a_indices, s1_d_indices = random.choice(np.array_split(_map, len(_map) // 3))
        assert len(s2_indices) == 1
        s2_index = s2_indices[0]

        # s1 images should have higher index in images list than s2 images
        assert all([i > s2_index for i in chain(s1_a_indices, s1_d_indices)])

        s2_patch = self.s2_transform(self.images[s2_index][:, i_slice, j_slice].astype(np.float32))
        s1_patch_ascending = self.images[random.choice(s1_a_indices)][:, i_slice, j_slice]
        s1_patch_descending = self.images[random.choice(s1_d_indices)][:, i_slice, j_slice]

        # stack s1 (either both or a random direction) and s2 patches
        if self.both_orbit_directions:
            s1_patch = torch.cat([self.s1_transform(p) for p in (s1_patch_ascending, s1_patch_descending)])
        else:
            s1_patch = self.s1_transform(random.choice([s1_patch_ascending, s1_patch_descending]))
        image_patch = torch.cat([s2_patch, s1_patch], dim=0)

        labels_patch = self.labels_transform(self.labels[labels_index][:, i_slice, j_slice])

        # assert nan labels are consistent throughout channel dimension
        assert (labels_patch.isnan() == labels_patch.isnan()[0]).all(), f"{i}, {j}, {labels_index}"

        return image_patch, labels_patch

    def __len__(self):
        return len(self.locations)


class RandomSampler(Sampler):
    """
    https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


class ToTensor(torch.nn.Module):
    """
    In contrast to torchvision.transforms.ToTensor, this class doesn't permute the images' dimensions.
    """

    def forward(self, array):
        return torch.from_numpy(array)


class SelectChannels(torch.nn.Module):
    """
    Selects specified channels from an [B, C, H, W] tensor, dropping all other channels.
    """

    def __init__(self, channels: np.array):
        super().__init__()
        self.channels = channels

    def forward(self, tensor):
        if tensor.ndim == 3:
            return tensor[self.channels, :, :]
        elif tensor.ndim == 4:
            return tensor[:, self.channels, :, :]
        raise ValueError(f'tensor.ndim should be in [3,4], got {tensor.ndim}')


class NoValDataException(Exception):
    pass
