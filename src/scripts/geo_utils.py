import os
from osgeo import gdal
import numpy as np
from skimage.transform import resize
from zipfile import ZipFile

GDAL_TYPE_LOOKUP = {
    'float32': gdal.GDT_Float32,
    'uint16': gdal.GDT_UInt16,
    'uint8': gdal.GDT_Byte
}


def sort_s2_band_arrays(band_arrays, channels_last=True):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def sort_s1_band_arrays(band_arrays, channels_last=True):
    bands = ['s1_asc_1', 's1_asc_2', 's1_desc_1', 's1_desc_2']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def get_tile_info(ref_dataset):
    return {
        'projection': ref_dataset.GetProjection(),
        'geotransform': ref_dataset.GetGeoTransform(),
        'width': ref_dataset.RasterXSize,
        'height': ref_dataset.RasterYSize
    }


def save_array_as_geotif(out_path, array, tile_info, out_type=None, dstnodata=None):
    if out_type is None:
        out_type = array.dtype.name
    out_type = GDAL_TYPE_LOOKUP[out_type]
    # PACKBITS is a lossless compression.
    # predictor=2 saves horizontal differences to previous value (useful for empty regions)
    dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, tile_info['width'], tile_info['height'], 1,
                                                  out_type, options=['COMPRESS=DEFLATE', 'PREDICTOR=2'])
    dst_ds.SetGeoTransform(tile_info['geotransform'])
    dst_ds.SetProjection(tile_info['projection'])

    dst_ds.GetRasterBand(1).WriteArray(array)
    if dstnodata is not None:
        dst_ds.GetRasterBand(1).SetNoDataValue(dstnodata)

    dst_ds.FlushCache()  # write to disk


def read_sentinel2_bands(data_path, from_aws=False, bucket='sentinel-s2-l2a', channels_last=False):
    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    if '.zip' in data_path:
        archive = ZipFile(data_path, 'r')  # data_path is path to zip file

    band_arrays = {}
    tile_info = None
    for res in bands_dir.keys():
        bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]

            if from_aws:
                print('Opening bands with gdal vsis3...')
                path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')
            else:
                # get datapath within zip file
                # get path to IMG_DATA
                path_img_data = [name for name in archive.namelist() if name.endswith('{}_{}m.jp2'.format(band_name, res))][0]
                path_band = os.path.join(data_path, path_img_data)
                path_band = '/vsizip/' + path_band

            print('path_band: ', path_band)
            ds = gdal.Open(path_band)
            if not tile_info:
                tile_info = get_tile_info(ds)

            # read all band data to memory once
            print('reading full band array...')
            band = ds.GetRasterBand(1)
            band_data = band.ReadAsArray()
            band_arrays[band_name] = band_data

    print("Opening CLD band...")
    path_img_data = [name for name in archive.namelist() if name.endswith('CLD_20m.jp2') or name.endswith('MSK_CLDPRB_20m.jp2')][0]
    path_band = os.path.join(data_path, path_img_data)
    path_band = '/vsizip/' + path_band
    print('cloud path_band:', path_band)
    ds = gdal.Open(path_band)
    print('reading full band array...')
    band = ds.GetRasterBand(1)
    band_arrays['CLD'] = band.ReadAsArray()

    target_shape = band_arrays['B02'].shape
    print('resizing 20m and 60m bands to 10m resolution...')
    for band_name in band_arrays:
        band_array = band_arrays[band_name]
        if band_array.shape != target_shape:
            band_arrays[band_name] = \
                resize(band_array, target_shape, mode='reflect', order=0, preserve_range=True).astype(np.uint16)

    print('sorting s2 bands...')
    image_array = sort_s2_band_arrays(band_arrays=band_arrays, channels_last=channels_last)
    return image_array, tile_info, band_arrays['SCL'], band_arrays['CLD']


def read_sentinel1_bands(s1_asc_path, s1_desc_path, channels_last=False):
    band_arrays = {}

    asc_image = gdal.Open(s1_asc_path)
    for i in range(1, 3):
        band_arrays[f's1_asc_{i}'] = asc_image.GetRasterBand(i).ReadAsArray()

    desc_image = gdal.Open(s1_desc_path)
    for i in range(1, 3):
        band_arrays[f's1_desc_{i}'] = desc_image.GetRasterBand(i).ReadAsArray()

    print('sorting s1 bands...')
    return sort_s1_band_arrays(band_arrays=band_arrays, channels_last=channels_last)
