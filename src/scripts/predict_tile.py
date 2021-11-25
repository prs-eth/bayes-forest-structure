# coding=utf-8
import sys
import yaml
import argparse
from pathlib import Path
import math

from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize
from torch.utils.data import Dataset, DataLoader

from geo_utils import save_array_as_geotif, read_sentinel2_bands, read_sentinel1_bands
sys.path.append('repo/src')
from models import ResNext
from data import ToTensor, SelectChannels


class Sentinel2Deploy(Dataset):
    """ Create patches to predict a full Sentinel-2 image. """

    def __init__(self, s2_path, s1_asc_path, s1_desc_path, s2_transforms, s1_transforms, patch_size=128,
                 border=8, from_aws=False):
        self.s2_path = s2_path
        self.s1_asc_path = s1_asc_path
        self.s1_desc_path = s1_desc_path
        self.from_aws = from_aws
        self.s2_transforms = s2_transforms
        self.s1_transforms = s1_transforms
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border

        self.s2_image, self.tile_info, self.scl, self.cloud = \
            read_sentinel2_bands(data_path=self.s2_path, from_aws=self.from_aws, channels_last=True)
        self.s1_image = read_sentinel1_bands(s1_asc_path, s1_desc_path, channels_last=True)

        self.image_shape_original = self.s2_image.shape
        # pad the images with channels in last dimension
        self.s2_image = np.pad(self.s2_image, ((self.border, self.border), (self.border, self.border), (0, 0)),
                               mode='symmetric')
        self.s1_image = np.pad(self.s1_image, ((self.border, self.border), (self.border, self.border), (0, 0)),
                               mode='symmetric')

        self.patch_coords_dict = self._get_patch_coords()
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        self.scl_exclude_labels = np.array([8, 9, 11])  # CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW
        self.scl = np.array(self.scl, dtype=np.uint8)

        print('image shape original: ', self.image_shape_original)
        print('after padding: image shapes: ', self.s2_image.shape, self.s1_image.shape)

    def _get_patch_coords(self):
        img_rows, img_cols = self.s2_image.shape[0:2]  # last dimension corresponds to channels

        print('img_rows, img_cols:', img_rows, img_cols)

        rows_tiles = int(math.ceil(img_rows / self.patch_size_no_border))
        cols_tiles = int(math.ceil(img_cols / self.patch_size_no_border))

        patch_coords_dict = {}
        patch_idx = 0
        for y in range(0, rows_tiles):
            y_coord = y * self.patch_size_no_border
            if y_coord > img_rows - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = img_rows - self.patch_size
            for x in range(0, cols_tiles):
                x_coord = x * self.patch_size_no_border
                if x_coord > img_cols - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = img_cols - self.patch_size
                patch_coords_dict[patch_idx] = {'x_topleft': x_coord,
                                                'y_topleft': y_coord}
                patch_idx += 1

        print('number of patches: ', len(patch_coords_dict))
        return patch_coords_dict

    def __getitem__(self, index):

        y_topleft = self.patch_coords_dict[index]['y_topleft']
        x_topleft = self.patch_coords_dict[index]['x_topleft']

        s2_patch = self.s2_image[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size, :]
        s1_patch = self.s1_image[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size, :]

        s2_patch = self.s2_transforms(s2_patch.astype('float32').transpose(2, 0, 1))
        s1_patch = self.s1_transforms(s1_patch.astype('float32').transpose(2, 0, 1))

        return torch.cat([s2_patch, s1_patch], dim=0)

    def __len__(self):
        return len(self.patch_coords_dict)

    def recompose_patches(self, patches, out_type=np.float32,
                          mask_empty=True, mask_negative=False,
                          mask_clouds=True, mask_with_scl=True, cloud_thresh_perc=5,
                          mask_tile_boundary=False):
        """ Recompose image patches or corresponding predictions to the full Sentinel-2 tile shape."""

        # init tile with channels first
        channels = patches.shape[1]
        height, width = self.s2_image.shape[:2]
        tile = np.full(shape=(channels, height, width), fill_value=np.nan, dtype=out_type)

        for index in range(len(patches)):
            y_topleft = self.patch_coords_dict[index]['y_topleft']
            x_topleft = self.patch_coords_dict[index]['x_topleft']

            tile[:, y_topleft+self.border:y_topleft + self.patch_size - self.border,
                 x_topleft+self.border:x_topleft + self.patch_size - self.border] \
                = patches[index, :,
                          self.border:self.patch_size - self.border,
                          self.border:self.patch_size - self.border]

        # remove padding to return original tile size
        tile = tile[:, self.border:-self.border, self.border:-self.border]

        # reduce first dimension if single band (e.g. predictions)
        tile = tile.squeeze()

        # masking
        tile_masked = tile
        if mask_empty:
            # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            # note self.image has shape: (height, width, channels)
            invalid_mask = (self.s2_image[self.border:-self.border, self.border:-self.border, 1:4] == 0).all(2) \
                | (self.s1_image[self.border:-self.border, self.border:-self.border] < 8.).all(2)
            print('number of empty pixels:', np.sum(invalid_mask))
            # mask empty image pixels
            tile_masked[:, invalid_mask] = np.nan

        if mask_negative:
            # mask negative values in the recomposed tile (e.g. predictions)
            tile_masked[tile_masked < 0] = np.nan

        if mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            tile_masked[:, np.isin(self.scl, self.scl_exclude_labels)] = np.nan

        if mask_clouds:
            tile_masked[:, self.cloud > cloud_thresh_perc] = np.nan

        if mask_tile_boundary:
            # top and bottom rows
            tile_masked[:, self.border] = np.nan
            tile_masked[:, -self.border:] = np.nan
            # left and right columns
            tile_masked[:, :, :self.border] = np.nan
            tile_masked[:, :, -self.border:] = np.nan

        return tile_masked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2_path")
    parser.add_argument("--s1_asc_path")
    parser.add_argument("--s1_desc_path")
    parser.add_argument("--config_path")
    parser.add_argument("--out_dir")
    parser.add_argument("--patch_size", default=512, help="Size of square patch (height=width)", type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--checkpoint_dirs", nargs="+")
    args, unknown = parser.parse_known_args()

    with open(args.config_path) as fh:
        train_config = yaml.safe_load(fh)

    stats = train_config['stats']
    s2_transforms = Compose([ToTensor(), Normalize(stats['s2_mean'], stats['s2_std'])])
    s1_transforms = Compose([ToTensor(), Normalize(stats['s1_mean'] * 2, stats['s1_std'] * 2)])

    ds_pred = Sentinel2Deploy(s2_path=args.s2_path,
                              s1_asc_path=args.s1_asc_path,
                              s1_desc_path=args.s1_asc_path,
                              s2_transforms=s2_transforms,
                              s1_transforms=s1_transforms,
                              patch_size=args.patch_size,
                              border=16)

    dl_pred = DataLoader(ds_pred, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         pin_memory=True)

    print('Initializing models...')
    num_orbit_directions = (2 if train_config['data']['both_orbit_directions'] else 1)
    num_s1_channels = len(train_config['data']['s1_image_bands']) * num_orbit_directions
    in_channels = len(train_config['data']['s2_image_bands']) + num_s1_channels
    out_channels = len(train_config['data']['labels_bands'])
    train_config['model'].pop('type')

    models = []
    for checkpoint_dir in args.checkpoint_dirs:
        model = ResNext(in_channels, out_channels, num_s1_channels=num_s1_channels, **train_config['model']).cuda()
        checkpoint_path = sorted(list((Path(checkpoint_dir) / 'checkpoints').glob('*.pt')))[-1]
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        models.append(model.eval())

    print(f'Loaded an ensemble of {len(models)} models')

    torch.set_grad_enabled(False)

    predictions = torch.full((len(ds_pred), out_channels, args.patch_size, args.patch_size), fill_value=np.nan)
    variances = torch.full((len(ds_pred), out_channels, args.patch_size, args.patch_size), fill_value=np.nan)
    variances_al = torch.full((len(ds_pred), out_channels, args.patch_size, args.patch_size), fill_value=np.nan)
    variances_ep = torch.full((len(ds_pred), out_channels, args.patch_size, args.patch_size), fill_value=np.nan)

    for step, inputs in enumerate(tqdm(dl_pred, ncols=100, desc='pred')):  # for each training step
        inputs = inputs.cuda()

        patch_means, patch_variances = [], []
        for model in models:
            patch_mean, patch_variance = model(inputs)
            if train_config['training']['activate_mean']:
                patch_mean[:, [0, 1]] = patch_mean[:, [0, 1]].exp()
                patch_mean[:, [2, 3, 4]] = torch.sigmoid(patch_mean[:, [2, 3, 4]])
            patch_means.append(patch_mean.detach().cpu())
            patch_variances.append(patch_variance.exp().detach().cpu())

        # calculate overall mean & variance for this patch
        overall_mean = torch.stack(patch_means).mean(0)
        overall_variance_al = torch.stack(patch_variances).mean(0)
        overall_variance_ep = (torch.stack(patch_means) - overall_mean).pow(2).mean(0)

        predictions[step * args.batch_size:(step + 1) * args.batch_size] = overall_mean
        variances_al[step * args.batch_size:(step + 1) * args.batch_size] = overall_variance_al
        variances_ep[step * args.batch_size:(step + 1) * args.batch_size] = overall_variance_ep
        variances[step * args.batch_size:(step + 1) * args.batch_size] = overall_variance_al + overall_variance_ep

    predictions = predictions.numpy()
    variances_al = variances_al.numpy()
    variances_ep = variances_ep.numpy()
    variances = variances.numpy()

    # recompose predictions and variances
    recomposed_tiles = {}
    for array, name in zip((predictions, variances_al, variances_ep, variances),
                           ('mean', 'variance_al', 'variance_ep', 'variance')):
        recomposed = ds_pred.recompose_patches(array, out_type=np.float32)
        print(f'recomposed tiles shape: {recomposed.shape}')

        for i, variable in enumerate(train_config['logging']['labels_names']):
            out_path = Path(args.out_dir) / variable / name / (Path(args.s2_path).stem + '.tif')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_array_as_geotif(str(out_path), recomposed[i], ds_pred.tile_info)
