import os
from pathlib import Path
import socket

import portalocker

def get_license_string():
    print('Waiting for lock...')
    with portalocker.Lock('license_list.txt', 'r+') as fh:
        licenses = [l.strip() for l in fh.readlines()]

        if len(licenses) == 0:
            return None

        print('Acquiring license', licenses[0])
        fh.seek(0)
        fh.write('\n'.join(licenses[1:]))
        fh.truncate()

    print('Released lock')

    return licenses[0]

hostname =  socket.gethostname()
license_path = Path('licenses') / hostname / 'gurobi.lic'
if not license_path.exists():
    license_string = get_license_string()

    if license_string is None:
        exit(1)

    os.system(f'grbgetkey {license_string} --path licenses/{hostname}')