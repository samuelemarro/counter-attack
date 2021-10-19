from pathlib import Path
import subprocess
import sys
sys.path.append('.')

import click
import jsonpickle

import batch_processing.batch_utils as batch_utils

TEST_NAMES = ['standard', 'adversarial', 'relu']

def run_and_output(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    return result.stdout.decode('utf-8')

def start_job(job):
    output = run_and_output(f'./run_job.sh {job.domain} {job.architecture} {job.test_name} {job.index}')
    prefix = 'Submitted batch job '
    assert output.startswith(prefix)
    job_number = output[len(prefix):]
    return int(job_number)

def create_job_list(config_dict):
    jobs = []

    configuration_priority = 0

    for domain, domain_config in config_dict.items():
        for architecture in domain_config['architectures']:
            for test_name in TEST_NAMES:
                # Domain + architecture + test_name = configuration
                for label, matching_indices in enumerate(domain_config['indices']):
                    for index_priority, index in enumerate(matching_indices):
                        # next_job will pick, in order:
                        # mnist a standard, label = 0, first element
                        # mnist a adversarial, label = 0, first element
                        # ...
                        # cifar10 c relu, label = 0, first element
                        # mnist a standard, label = 1, first element
                        # ...
                        # ...
                        # cifar10 c relu, label = 9, first element
                        # mnist a standard, label = 0, second element
                        # and so on
                        priority = (index_priority, label, configuration_priority)
                        job = batch_utils.Job(domain, architecture, test_name, index, priority=priority)

                        jobs.append(job)

                configuration_priority += 1
    
    return jobs

def write_queue_job(job, tracker_path):
    with open(tracker_path, 'a') as f:
        f.write(f'QUEUED,{job.to_csv()}\n')

def register_job(job, tracker_path):
    with open(tracker_path, 'a') as f:
        f.write(f'REGISTERED,{job.to_csv()},{job.job_number}\n')

@click.command()
@click.argument('tracker_path', type=click.Path(dir_okay=False, file_okay=True))
@click.argument('config_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('job_count', type=int)
@click.option('--new', is_flag=True)
@click.option('--update', is_flag=True)
def main(tracker_path, config_file, job_count, new, update):
    if new and update:
        raise click.UsageError('--new and --update are mutually exclusive')

    if new or update:
        with open(config_file) as f:
            config_dict = jsonpickle.decode(f.read())
    else:
        raise click.UsageError('Choose one between --new and --update')

    if new:
        if Path(tracker_path).exists():
            raise click.BadArgumentUsage('tracker_path must not exist.')
        new_jobs = create_job_list(config_dict)
        actual_job_count = job_count
    else:
        new_jobs = []

        current_job_ids = batch_utils.read_jobs(tracker_path).keys()
        print(current_job_ids)
        
        for job in create_job_list(config_dict):
            if job.unique_id not in current_job_ids:
                new_jobs.append(job)

        actual_job_count = job_count - len(current_job_ids)
    print(len(new_jobs),'new jobs')
    print(f'Starting {actual_job_count} jobs.')
    jobs_by_priority = sorted(new_jobs, key=lambda x: x.priority)

    jobs_to_start = jobs_by_priority[:actual_job_count]

    for job in jobs_to_start:
        print('Starting job', job)
        write_queue_job(job, tracker_path)
        job_number = start_job(job)
        job.job_number = job_number
        register_job(job, tracker_path)

if __name__ == '__main__':
    main()