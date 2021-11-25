import re
import wget
from urllib.error import HTTPError

BASE_URL = 'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/'
SAVE_PATH = '/scratch2/albecker/downloads/DEM'

for N in range(58, 72):
	for E in range(3, 32):
		name = f'Copernicus_DSM_COG_10_N{N:02}_00_E{E:03}_00_DEM'
		try:
			wget.download(f'{BASE_URL}{name}/{name}.tif', out=SAVE_PATH)
		except HTTPError as error:
			if error.code != 404:
				raise
