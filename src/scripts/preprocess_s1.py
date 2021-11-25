import os
import sys
from pathlib import Path

GRAPH_FILE = '/home/pf/pfstaff/projects/albecker_forest_structure/repo/config/process_s1.xml'
OUT_NAME = 'preprocessed.tif'
OVERWRITE = True

failed = []

for safe_dir in sys.argv[1:]:
    safe_dir = Path(safe_dir)
    print(f'Processing {safe_dir.stem}...')

    if (safe_dir / OUT_NAME).exists() and not OVERWRITE:
        print('Exists, skipping...')
        continue

    command = f'gpt {GRAPH_FILE} -Ptarget="{safe_dir / OUT_NAME}" -Dsnap.dataio.bigtiff.compression.type=LZW {safe_dir}'
    if os.system(command) != 0:
        # remove target file and exit
        (safe_dir / OUT_NAME).unlink(missing_ok=True)
        failed.append(safe_dir.stem)

print('Failed:\n' + '\n'.join(failed))
