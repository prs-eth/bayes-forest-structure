import sys
import random
import yaml
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from itertools import chain

import torch
import numpy as np
import fiona
import rasterio.warp
import rasterio.features
from torchvision.transforms import Compose, Normalize
from blowtorch import Run

sys.path.append('repo/src')
from models import ResNext
from data import ToTensor, SelectChannels

SEPARATOR = 65535


def parse_date(date_str) -> datetime:
    return datetime.strptime(date_str, '%Y%m%d')


run = Run(config_files=['repo/config/predict_testset.yaml'])
run.seed_all(12345)

save_dir = Path('results/') / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + (f'_{run["name"]}' if run['name'] else ''))
save_dir.mkdir()

assert run['patch_size'] % 2 == 1, 'Patch size should be odd.'

with (Path(run['checkpoint_dirs'][0]) / 'config.yaml').open() as fh:
    train_config = yaml.safe_load(fh)

with (Path(run['pkl_dir']) / 'stats.yaml').open() as fh:
    # load training set statistics for data normalization
    stats = yaml.safe_load(fh)

s2_channels = (np.array(train_config['data']['s2_image_bands']) - 1).astype('int')
s1_channels = (np.array(train_config['data']['s1_image_bands']) - 1).astype('int')
s2_mean = np.array(stats['s2_mean'])[s2_channels].tolist()
s2_std = np.array(stats['s2_std'])[s2_channels].tolist()
s1_mean = np.array(stats['s1_mean'])[s1_channels].tolist()
s1_std = np.array(stats['s1_std'])[s1_channels].tolist()
s2_transform = Compose([ToTensor(), SelectChannels(s2_channels), Normalize(s2_mean, s2_std)])
s1_transform = Compose([ToTensor(), SelectChannels(s1_channels), Normalize(s1_mean, s1_std)])

print('Initializing models...')
num_orbit_directions = (2 if train_config['data']['both_orbit_directions'] else 1)
num_s1_channels = len(train_config['data']['s1_image_bands']) * num_orbit_directions
in_channels = len(train_config['data']['s2_image_bands']) + num_s1_channels
out_channels = len(train_config['data']['labels_bands'])
model_type = train_config['model'].pop('type')

models = []
for checkpoint_dir in run['checkpoint_dirs']:
    model = ResNext(in_channels, out_channels, num_s1_channels=num_s1_channels, **train_config['model']).cuda()
    checkpoint_path = sorted(list((Path(checkpoint_dir) / 'checkpoints').glob('*.pt')))[-1]
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    models.append(model.eval())

print(f'Loaded an ensemble of {len(models)} models')

torch.set_grad_enabled(False)

num_predicted_pixels = 0
num_nans_in_polygon = 0

for gt_file_path in Path(run['gt_dir']).glob('*.tif'):
    if gt_file_path.stem not in run['projects']:
        continue

    gt_file = rasterio.open(gt_file_path)
    gt_mask = gt_file.read_masks(1).astype(np.bool)

    with rasterio.open(Path(run['split_mask_dir']) / gt_file_path.name) as split_file:
        split_mask = split_file.read(1).astype('float16')

    # reproject project polygon to labels crs and rasterize
    polygon, crs = None, None
    for collection in [fiona.open(p) for p in run['project_shapefiles']]:
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

    patch_half = run['patch_size'] // 2
    margin = run['margin']
    step_size = run['patch_size'] - (2 * margin)

    num_images_per_pixel = np.zeros((1, *gt_file.shape), dtype=np.uint8)
    mean = np.full((5, *gt_file.shape), np.nan, dtype=np.float32)
    variance = np.full((5, *gt_file.shape), np.nan, np.float32)

    # locate all test stripes within the project region
    stripes = []
    current_start = None
    for i in range(gt_file.shape[0]):
        if split_mask[i, 0] == 2. and current_start is None:
            current_start = i
        elif split_mask[i, 0] != 2. and current_start is not None:
            stripes.append((current_start, i))
            current_start = None
    if current_start is not None:
        stripes.append((current_start, gt_file.shape[0]))

    for start, end in stripes:
        if end - start < 15:
            # not enough pixels to carry out one sweep, skip stripe
            continue

        print(f'Processing stripe [{start}, {end})')

        for i in chain(range(start + patch_half, end - patch_half, step_size), [end - patch_half - 1]):
            for j in chain(range(patch_half, gt_file.shape[1] - patch_half, step_size),
                           [gt_file.shape[1] - patch_half - 1]):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)

                # assert only test pixels
                assert (split_mask[i_slice, j_slice] == 2.).all()

                if (rasterized_polygon[i_slice, j_slice] == 0.).all():
                    # fully outside project polygon, skip patch
                    continue

                if (~gt_mask[i_slice, j_slice]).all():
                    # no valid test points in patch, skip
                    continue

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

                patch_means, patch_variances = [], []
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

                    # determine matching s1 image date. All S1 images within 15 days of the S2 image will be
                    # queried and the first one is selected.
                    try:
                        matching_ascending = [img for img, date in valid_ascending if
                                              abs((s2_date - date).days) <= 15][0]
                        matching_descending = [img for img, date in valid_descending if
                                               abs((s2_date - date).days) <= 15][0]
                    except IndexError:
                        # no matching ascending and descending image found for this s2 image and patch
                        continue

                    s2_dates_used.add(s2_date)

                    # construct model input
                    s2 = s2_transform(s2_image[:, i_slice, j_slice].astype('float32'))
                    s1_asc = s1_transform(matching_ascending[:, i_slice, j_slice].astype('float32'))
                    s1_desc = s1_transform(matching_descending[:, i_slice, j_slice].astype('float32'))
                    if train_config['data']['both_orbit_directions']:
                        _input = torch.cat([s2, s1_asc, s1_desc], dim=0).unsqueeze(0).cuda()
                    else:
                        _input = torch.cat([s2, random.choice([s1_asc, s1_desc])], dim=0).unsqueeze(0).cuda()

                    for model in models:
                        patch_mean, patch_variance = model(_input)
                        if train_config['training']['activate_mean']:
                            patch_mean[:, [0, 1]] = patch_mean[:, [0, 1]].exp()
                            patch_mean[:, [2, 3, 4]] = torch.sigmoid(patch_mean[:, [2, 3, 4]])
                        patch_variance = patch_variance.exp()
                        patch_means.append(patch_mean.detach().cpu()[:, :, margin:-margin, margin:-margin])
                        patch_variances.append(patch_variance.detach().cpu()[:, :, margin:-margin, margin:-margin])

                    break

                if len(patch_means):
                    # slices spanning only the inner part of the patch, excluding margins
                    i_slice_inner = slice(i - (step_size // 2), i + (step_size // 2) + 1)
                    j_slice_inner = slice(j - (step_size // 2), j + (step_size // 2) + 1)
                    num_images_per_pixel[0, i_slice_inner, j_slice_inner] = len(patch_means) // len(models)

                    # calculate overall mean & variance for this patch
                    mean[:, i_slice_inner, j_slice_inner] = torch.cat(patch_means).mean(0)
                    variance[:, i_slice_inner, j_slice_inner] = torch.cat(patch_variances).mean(0) + \
                        (torch.cat(patch_means) - mean[:, i_slice_inner, j_slice_inner]).pow(2).mean(0)

    # set pixels to nan that lie outside the project polygon
    mean[:, rasterized_polygon == 0.] = np.nan
    variance[:, rasterized_polygon == 0.] = np.nan

    # set pixels to nan that are not valid in the ground truth
    mean[:, ~gt_mask] = np.nan
    variance[:, ~gt_mask] = np.nan

    stats = {
        'num_predicted_pixels': ((rasterized_polygon == 1.) & ~np.isnan(mean).all(0)).sum().item(),
        'num_nans_in_polygon': ((rasterized_polygon == 1.) & (split_mask == 2.) & np.isnan(mean).all(0)).sum().item()
    }
    num_predicted_pixels += stats['num_predicted_pixels']
    num_nans_in_polygon += stats['num_nans_in_polygon']

    print('Writing results...')

    with (save_dir / f'{gt_file_path.stem}_stats.txt').open('w') as fh:
        yaml.dump(stats, fh)

    with rasterio.Env():
        profile = gt_file.profile
        profile.update(
            driver='GTiff',
            count=1,
            compress='deflate',
            nodata=None,
            dtype='uint8'
        )
        with rasterio.open(save_dir / f'{gt_file_path.stem}_num_images.tif', 'w', **profile) as f:
            f.write(num_images_per_pixel)

    with rasterio.Env():
        profile = gt_file.profile
        profile.update(
            driver='GTiff',
            count=5,
            compress='deflate',
            nodata=None,
            dtype='float32'
        )
        with rasterio.open(save_dir / f'{gt_file_path.stem}_mean.tif', 'w', **profile) as f:
            f.write(mean)

    with rasterio.Env():
        profile = gt_file.profile
        profile.update(
            driver='GTiff',
            count=5,
            compress='deflate',
            nodata=None,
            dtype='float32'
        )
        with rasterio.open(save_dir / f'{gt_file_path.stem}_variance.tif', 'w', **profile) as f:
            f.write(variance)

    gt_file.close()

# save config file along with the pickle objects
with open(save_dir / 'prediction_config.yaml', 'w') as fh:
    yaml.dump(run.get_raw_config(), fh)

# save overall prediction stats
with (save_dir / 'stats.txt').open('w') as fh:
    yaml.dump({'num_predicted_pixels': num_predicted_pixels, 'num_nans_in_polygon': num_nans_in_polygon}, fh)
