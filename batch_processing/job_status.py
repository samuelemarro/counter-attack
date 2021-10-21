from pathlib import Path
import sys
sys.path.append('.')

import click

import batch_processing.batch_utils as batch_utils

@click.command()
@click.argument('tracker_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
def main(tracker_path):
    current_jobs = batch_utils.read_jobs(tracker_path).values()

    queued = []
    running = []
    completed = []
    failed = []

    for job in current_jobs:
        if job.status == 'QUEUED':
            queued.append(job)
        elif job.status == 'RUNNING':
            running.append(job)
        elif job.status == 'FINISHED':
            # A finished job might have actually failed
            results_path = Path('mip_results') / job.test_name / (job.domain + '-' + job.architecture) / f'{job.index}-{job.index + 1}.zip'
            if results_path.exists():
                # Success
                completed.append(job)
            else:
                # Failure
                failed.append(job)
        else:
            raise NotImplementedError(f'Unsupported status "{job.status}"')

    print('=========')
    print('Queued:')
    for job in queued:
        print(job)

    print('=========')
    print('Completed:')
    for job in completed:
        print(job)

    print('=========')
    print('Running:')
    for job in running:
        print(job)

    print('=========')
    print('Failed:')
    for job in failed:
        print(job)

    print('=========')
    print('Queued:', len(queued))
    print('Running:', len(running))
    print('Completed:', len(completed))
    print('Failed:', len(failed))

if __name__ == '__main__':
    main()