import os
from pathlib import Path
import socket

import click
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

@click.command()
@click.argument('--server', type=str, default=None)
def main(server):
    base_folder = Path('licenses')
    base_folder.mkdir(exist_ok=True)

    hostname =  socket.gethostname()

    license_path = base_folder / hostname / 'gurobi.lic'
    if not license_path.exists():
        license_string = get_license_string()

        if license_string is None:
            exit(1)

        command = f'grbgetkey {license_string} --verbose --path licenses/{hostname} '
        if server is not None:
            command += f'--server {server}'

        os.system(command)

if __name__ == '__main__':
    main()