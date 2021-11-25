"""
Preprocesses the NIBIO forest structure dataset and pickles train/val data lists.
"""

from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from itertools import chain
from collections import defaultdict
import pickle
import yaml

from blowtorch import Run
from tqdm import trange
import numpy as np
import fiona
import rasterio.warp
import rasterio.features


SEPARATOR = 65535


def parse_date(date_str) -> datetime:
    return datetime.strptime(date_str, '%Y%m%d')


run = Run(config_files=['repo/config/prepare_dataset.yaml'])
run.seed_all(12345)

save_dir = Path('data/forest_structure/pkl/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
save_dir.mkdir(parents=True)

assert run['patch_size'] % 2 == 1, 'Patch size should be odd.'

# shapefile that describes the acquisition projects' polygons
project_shape_collections = [fiona.open(p) for p in run['project_shapefiles']]

num_train, num_val, num_test = 0, 0, 0

for gt_file_path in Path(run['gt_dir']).glob('*.tif'):
    if gt_file_path.stem not in run['projects']:
        continue

    locations = defaultdict(list)
    loc_to_images_map = defaultdict(list)
    offsets = defaultdict(list)

    gt_file = rasterio.open(gt_file_path)
    valid_mask = gt_file.read_masks(1)
    labels = gt_file.read(run['data_bands'])
    # set invalid gt points to nan
    labels[:, valid_mask == 0] = np.nan

    with rasterio.open(Path(run['split_mask_dir']) / gt_file_path.name) as split_file:
        split_mask = split_file.read(1).astype('float16')

    # reproject project polygon to labels crs and rasterize
    polygon, crs = None, None
    for collection in project_shape_collections:
        try:
            polygon = [s['geometry'] for s in collection if s['properties']['kv_id'] == int(gt_file_path.stem)][0]
            crs = collection.crs
            break
        except IndexError:
            pass
    polygon = rasterio.warp.transform_geom(crs, gt_file.crs, polygon)
    rasterized_polygon = rasterio.features.rasterize(
        [(polygon, 1)],
        out_shape=gt_file.shape,
        transform=gt_file.transform,
        fill=0,
        dtype='uint8'
    )

    print(f'Reading in images for {gt_file_path.name}...')
    s2_images: [Tuple[np.ndarray, datetime]] = []
    s1_images_ascending: [Tuple[np.ndarray, datetime]] = []
    s1_images_descending: [Tuple[np.ndarray, datetime]] = []

    for img_path in (Path(run['s2_reprojected_dir']) / gt_file_path.stem).glob('*.tif'):
        with rasterio.open(img_path) as fh:
            s2_images.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[3].split('T')[0])))

    for img_path in (Path(run['s1_reprojected_dir']) / gt_file_path.stem).glob('*.tif'):
        with rasterio.open(img_path) as fh:
            if img_path.stem.endswith('_A'):
                s1_list = s1_images_ascending
            elif img_path.stem.endswith('_D'):
                s1_list = s1_images_descending
            else:
                raise ValueError(f'Could not extract orbit direction from filename: {img_path.name}')
            s1_list.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[5].split('T')[0])))

    # helpful for later dataset encoding
    images = [img for img, _ in chain(s2_images, s1_images_ascending, s1_images_descending)]
    image_ids = [id(img) for img in images]

    num_images_per_pixel = np.zeros((1, gt_file.shape[0], gt_file.shape[1]), dtype=np.uint8)

    if run['sampling_strategy'] == 'valid_center':
        patch_half = run['patch_size'] // 2
        for i in trange(patch_half, gt_file.shape[0] - patch_half):
            for j in range(patch_half, gt_file.shape[1] - patch_half):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)
                is_same_split = (split_mask[i_slice, j_slice] == split_mask[i_slice, j_slice][0, 0]).all()
                is_in_polygon = (rasterized_polygon[i_slice, j_slice] == 1).all()

                # add patches that have a valid center pixel, only consist of one split class and lie completely
                # within their respective project polygon
                if valid_mask[i, j] and is_same_split and is_in_polygon:
                    images_for_pixel: List[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]] = []
                    s2_dates_used = set()

                    # filter out s1 images which contain a nodata pixel in the patch, i.e. images which do
                    # not fully cover the patch. We noticed that some s1 images have weird stripes with
                    # values close to (but not exactly) zero near the swath edge. Therefore we empirically
                    # set the threshold value to 8.
                    valid_ascending = [img for img in s1_images_ascending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    valid_descending = [img for img in s1_images_descending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    if len(valid_ascending) == 0 or len(valid_descending) == 0:
                        continue

                    for s2_image, s2_date in s2_images:
                        # do not add image if an image of the same date has been added for this location before.
                        # this is the case e.g. for the overlap region between two adjacent S2 images, which is
                        # identical for both images and would result in duplicate data points.
                        if s2_date in s2_dates_used:
                            continue

                        # only add patch if there is no nodata pixel contained, where a nodata pixel is
                        # defined as having zeros across all channels.
                        if (s2_image[:, i_slice, j_slice] == 0.).all(0).any():
                            continue

                        # only add patch with less than 10% cloudy pixels, where a cloudy pixel is defined as
                        # having cloud probability > 10%.
                        if (s2_image[-1, i_slice, j_slice] > run['cloud_prob_threshold']).sum() \
                                / run['patch_size']**2 > run['cloudy_pixels_threshold']:
                            continue

                        # determine matching s1 image date(s). All S1 images within 15 days of the S2 image will be
                        # added (and sampled randomly from during training).
                        matching_ascending = [img for img, date in valid_ascending if
                                              abs((s2_date - date).days) <= 15]
                        matching_descending = [img for img, date in valid_descending if
                                               abs((s2_date - date).days) <= 15]

                        # add s2 and matching s1 images to list of available images for this location
                        if len(matching_ascending) and len(matching_descending):
                            images_for_pixel.append((s2_image, matching_ascending, matching_descending))
                            s2_dates_used.add(s2_date)

                    num_images_per_pixel[0, i, j] = len(images_for_pixel)

                    # a data point corresponds to one image coordinate, such that regions with higher number
                    # of available images are not oversampled during training. Only add if there's at least one image
                    # for that pixel.
                    if len(images_for_pixel):
                        data_point = (i, j, images_for_pixel, labels)

                        # transform `images_for_pixel` into contiguos numpy array where images are referenced based on
                        # their index in `images`
                        this_loc_to_images_map = []
                        for s2_image, s1_a_list, s1_d_list in images_for_pixel:
                            this_loc_to_images_map.extend(
                                [image_ids.index(id(s2_image)), SEPARATOR]
                                + [image_ids.index(id(img)) for img in s1_a_list]
                                + [SEPARATOR]
                                + [image_ids.index(id(img)) for img in s1_d_list]
                                + [SEPARATOR]
                            )

                        if split_mask[i, j] == 0:
                            dataset = 'train'
                        elif split_mask[i, j] == 1:
                            dataset = 'val'
                        elif split_mask[i, j] == 2:
                            dataset = 'test'
                        else:
                            raise ValueError(split_mask[i, j])

                        locations[dataset].append((i, j))
                        offsets[dataset].append(len(loc_to_images_map[dataset]))
                        loc_to_images_map[dataset].extend(this_loc_to_images_map)

    else:
        raise NotImplementedError(f'{run["sampling_strategy"]} not implemented.')

    # Packing data in contiguous numpy arrays alleviates a "memory leak" problem connected to Python multiprocessing
    # (see https://github.com/pytorch/pytorch/issues/13246)
    with open(save_dir / f'{gt_file_path.stem}.pkl', 'wb') as fh:
        pickle.dump({
            'images': images,
            'train': (
                np.array(locations['train'], dtype=np.uint16),
                np.array(loc_to_images_map['train'], dtype=np.uint16),
                np.array(offsets['train'], dtype=np.uint64)
            ),
            'val': (
                np.array(locations['val'], dtype=np.uint16),
                np.array(loc_to_images_map['val'], dtype=np.uint16),
                np.array(offsets['val'], dtype=np.uint64)
            ),
            'test': (
                np.array(locations['test'], dtype=np.uint16),
                np.array(loc_to_images_map['test'], dtype=np.uint16),
                np.array(offsets['test'], dtype=np.uint64)
            ),
            'labels': labels
        }, fh)

    num_train += len(locations['train'])
    num_val += len(locations['val'])
    num_test += len(locations['test'])

    # save num_images_per_pixel file
    with rasterio.Env():
        profile = gt_file.profile
        profile.update(
            driver='GTiff',
            count=1,
            compress='deflate',
            nodata=None,
            dtype='uint8'
        )
        with rasterio.open(save_dir / f'num_images_per_pixel_{gt_file_path.stem}.tif', 'w', **profile) as f:
            f.write(num_images_per_pixel)

    gt_file.close()

print(f'Train/val/test pixels: {num_train}/{num_val}/{num_test}')

# save data set cardinalities along with the pickle objects
with open(save_dir / 'stats.yaml', 'w') as fh:
    yaml.dump({'num_train': num_train, 'num_val': num_val, 'num_test': num_test}, fh)

# save config file along with the pickle objects
with open(save_dir / 'data_config.yaml', 'w') as fh:
    yaml.dump(run.get_raw_config(), fh)
