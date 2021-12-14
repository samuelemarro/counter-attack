from pathlib import Path
import shutil
import sys
sys.path.append('.')

import click

import batch_processing.batch_utils as batch_utils

@click.command()
@click.argument('tracker_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument('action', type=click.Choice(['backup', 'delete_logs', 'clean_tracker']))
@click.option('--backup-dir', type=click.Path(dir_okay=True, file_okay=False, exists=False), default=None)
@click.option('--new-tracker-path', type=click.Path(dir_okay=False, file_okay=True, exists=False), default=None)
@click.option('--dry-run', is_flag=True)
def main(tracker_path, action, backup_dir, new_tracker_path, dry_run):
    if action in ['backup', 'delete_logs'] and backup_dir is None:
        raise click.UsageError('You must specify a backup directory with --backup-dir.')

    if action == 'clean_tracker':
        if new_tracker_path is None:
            raise click.UsageError('--new-tracker-path is required when using "clean_tracker".')
        if Path(new_tracker_path).exists():
            raise click.BadOptionUsage('--new-tracker-path', f'{new_tracker_path} already exists.')

    copy_orders = []
    delete_orders = []
    tracker_delete_orders = []

    current_jobs = batch_utils.read_jobs(tracker_path).values()

    for job in current_jobs:
        if job.status == 'FINISHED':
            # A finished job might have actually failed
            subpath = Path(job.test_name) / (job.domain + '-' + job.architecture) / f'{job.index}-{job.index + 1}'

            results_path = Path('mip_results') / subpath.with_suffix('.zip')

            if not results_path.exists():
                # Failure

                global_logs_path = Path('global_logs') / subpath.with_suffix('.log')
                backup_logs_global_path = Path('backup') / 'global_logs' / subpath.with_suffix('.log')

                logs_dir_path = Path('logs') / subpath
                backup_logs_dir_path = Path('backup') / 'logs' / subpath

                if action == 'backup':
                    if global_logs_path.exists():
                        if backup_logs_global_path.exists():
                            raise click.BadOptionUsage('--backup-dir', f'{backup_logs_global_path} already exists.')

                        copy_orders.append((global_logs_path, backup_logs_global_path))
                    if logs_dir_path.exists():
                        if backup_logs_dir_path.exists():
                            raise click.BadOptionUsage('--backup-dir', f'{backup_logs_dir_path} already exists.')

                        copy_orders.append((logs_dir_path, backup_logs_dir_path))
                elif action == 'delete_logs':
                    if global_logs_path.exists():
                        if not backup_logs_global_path.exists():
                            raise RuntimeError(f'Attempting to delete file {global_logs_path} which has not been backed up.')
                        delete_orders.append(global_logs_path)

                    if logs_dir_path.exists():
                        if not backup_logs_dir_path.exists():
                            raise RuntimeError(f'Attempting to delete directory {logs_dir_path} which has not been backed up.')
                        delete_orders.append(logs_dir_path)

                elif action == 'clean_tracker':
                    if (global_logs_path.exists() and not backup_logs_global_path.exists()) or (logs_dir_path.exists() and not backup_logs_dir_path.exists()):
                        raise RuntimeError(f'Attempting to clean tracker history of file {job} which has not been fully backed up.')
                    tracker_delete_orders.append(job)
    
    if action == 'backup':
        print('Backing up files...')
        for src, dst in copy_orders:
            print(src)
            if not dry_run:
                shutil.copy2(src, dst)
    elif action == 'delete_logs':
        print('The following logs will be deleted:')
        for path in delete_orders:
            print(path)
        print('Type "delete_logs" to continue.')
        if input() != 'delete_logs':
            print('Aborted.')
            return
        print('Deleting logs...')
        for path in delete_orders:
            if not dry_run:
                path.unlink()
    elif action == 'clean_tracker':
        print('Creating a tracker without the following jobs:')
        for job in tracker_delete_orders:
            print(job)
        if not dry_run:
            batch_utils.remove_jobs(tracker_path, tracker_delete_orders, new_tracker_path)

    print('Done.')

if __name__ == '__main__':
    main()