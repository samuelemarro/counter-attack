from pathlib import Path
import sys
sys.path.append('.')

import click

import batch_processing.batch_utils as batch_utils

@click.command()
@click.argument('tracker_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
def main(tracker_path):
    current_jobs = batch_utils.read_jobs(tracker_path).values()

    counts = {
        'queued' : 0,
        'running' : 0,
        'completed' : 0,
        'failed' : 0
    }

    failures = []

    for job in current_jobs:
        if job.status == 'QUEUED':
            counts['queued'] += 1
        elif job.status == 'RUNNING':
            counts['running'] += 1
        elif job.status == 'FINISHED':
            # A finished job might have actually failed
            results_path = Path('mip_results') / job.test_name / (job.domain + '-' + job.architecture) / f'{job.index}-{job.index + 1}.zip'
            if results_path.exists():
                # Success
                counts['completed'] += 1
            else:
                # Failure
                counts['failed'] += 1
                failures.append(job)
        else:
            raise NotImplementedError(f'Unsupported status "{job.status}"')
    
    print('=========')
    print('Queued:', counts['queued'])
    print('Running:', counts['running'])
    print('Completed:', counts['completed'])
    print('Failed:', counts['failed'])
    print('=========')
    
    if len(failures) > 0:
        print('Failures:')
        for job in current_jobs:
            print(job)

if __name__ == '__main__':
    main()