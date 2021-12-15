import os
from pathlib import Path
import random
import socket
import time

RETRY_LIMIT = 20

import click
import portalocker

def register_license(hostname, server, license_string):
    command = f'grbgetkey {license_string} --verbose --path licenses/{hostname} '
    if server is not None:
        command += f'--server {server}'

    os.system(command)

def retrieve_license(hostname, server, license_path):
    print('Waiting for lock...')

    with portalocker.Lock('license_list.txt', 'r+') as fh:
        print('Lock acquired.')

        # Check that the license hasn't already been registered during waiting
        if not license_path.exists():
            # If the license has not been registered, register it
            licenses = [l.strip() for l in fh.readlines()]

            if len(licenses) == 0:
                # No licenses, abort
                exit(1)

            license = licenses[0]

            print('Acquired license', license)

            print('Registering...')
            register_license(hostname, server, license)
            print('Registered license', license)

            fh.seek(0)
            fh.write('\n'.join(licenses[1:]))
            fh.truncate()

        print('Released lock')

@click.command()
@click.option('--server', type=str, default=None)
def main(server):
    base_folder = Path('licenses')
    base_folder.mkdir(exist_ok=True)

    hostname =  socket.gethostname()

    license_path = base_folder / hostname / 'gurobi.lic'

    if not license_path.exists():
        print('No license found, registering...')

        success = False

        # Retrieving the license sometimes fails, so we retry a few times
        for i in range(RETRY_LIMIT):
            try:
                retrieve_license(hostname, server, license_path)
                success = True
                break
            except portalocker.LockException:
                print('Retrieval failed with exception, retrying...')
                time.sleep(random.randint(2 ** i, 2 ** (i + 1)))

        if not success:
            print('Max attempts reached, aborting.')
            exit(1)

if __name__ == '__main__':
    main()