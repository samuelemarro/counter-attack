import os
from random import random
import subprocess
import time

import click
import jsonpickle
from jsonpickle.pickler import encode

TEST_NAMES = ['standard', 'adversarial', 'relu']

class Job:
    def __init__(self, test_name, domain, architecture, index):
        self.test_name = test_name
        self.domain = domain
        self.architecture = architecture
        self.index = index
        self.job_number = None
    
    def __str__(self) -> str:
        return f'{self.test_name} {self.domain} {self.architecture} {self.index} (job number: {self.job_number})'
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, Job):
            return self.test_name == self.test_name and self.domain == o.domain and \
            self.architecture == o.architecture and self.index == o.index #and self.job_number == o.job_number
        return False

    def __hash__(self) -> int:
        return hash(self.index)

def run_and_output(command):
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def get_job_info(job_id):
    return run_and_output(f'scontrol show job {job_id}')

def parse_job_info(job_info):
    information = {}

    for pair in job_info.split(' '):
        split_pair = pair.split('=')

        if len(split_pair) == 2:
            information[split_pair[0]] = split_pair[1]

    return information

def get_job_status(job_id):
    from random import random
    return random() < 0.1
    info = parse_job_info(get_job_info(job_id))
    return info['JobState'] == 'COMPLETED'

def next_job(ready_jobs):
    def stringify(job):
        return job.test_name + '-' + job.domain + '-' + job.architecture
    
    # Split by configuration
    split_jobs = {}

    for job in ready_jobs:
        configuration_string = stringify(job)
        if configuration_string not in split_jobs:
            split_jobs[configuration_string] = []
        split_jobs[configuration_string].append(job)

    # Select the longest job list
    job_lists_by_length = sorted(list(split_jobs.items()), key=lambda x: len(x[1]))
    _, selected_jobs = job_lists_by_length[-1]

    # Select the one with the lowest index
    selected_job = sorted(selected_jobs, key=lambda x: x.index)[0]

    return selected_job

def run_job(job):
    from random import randint
    return randint(0, 10000)
    output = run_and_output(f'./start_job.sh {job.test_name} {job.domain} {job.architecture} {job.index}')
    prefix = 'Submitted batch job '
    assert output.startswith(prefix)
    job_number = output[len(prefix):]
    return int(job_number)

def create_job_list(config_dict):
    jobs = []

    for test_name in TEST_NAMES:
        for domain, domain_config in config_dict.items():
            for architecture in domain_config['architectures']:
                for index in domain_config['indices']:
                    job = Job(test_name, domain, architecture, index)

                    jobs.append(job)
    
    return jobs

@click.command()
@click.argument('tracker_path', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--new', is_flag=True)
@click.option('--update', is_flag=True)
@click.option('--tick', type=float, default=1)
@click.option('--job-cap', type=int, default=10)
@click.option('--config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
def main(tracker_path, new, update, tick, job_cap, config_file):
    # --merge: cerca quelli nuovi a partire dalla configurazione, li aggiunge
    # TODO: Permettere agli script di specificare la root folder dei risultati

    ready_jobs = []
    running_jobs = []
    completed_jobs = []

    def update_tracker():
        with open(tracker_path, 'w') as f:
            f.write(jsonpickle.encode({
                'ready' : ready_jobs,
                'running' : running_jobs,
                'completed' : completed_jobs
            }, f))
    
    if new and update:
        raise click.UsageError('--new and --update are mutually exclusive')
    
    if config_file is None:
            raise click.UsageError('Please specify a config file.')

    if new or update:
        with open(config_file) as f:
            config_dict = jsonpickle.decode(f.read())

    if new:
        ready_jobs = create_job_list(config_dict)
    else:
        with open(tracker_path, 'r') as f:
            tracker_status = jsonpickle.decode(f.read())
            ready_jobs = tracker_status['ready']
            running_jobs = tracker_status['running']
            completed_jobs = tracker_status['completed']
        
        if update:
            for job in create_job_list(config_dict):
                if job not in ready_jobs and job not in running_jobs and job not in completed_jobs:
                    ready_jobs.append(job)


    def check_jobs():
        new_completed_jobs = []
        print('===')
        print('Current jobs:', len(running_jobs))
        print('Ready jobs:', len(ready_jobs))
        print('Completed jobs:', len(completed_jobs))
        print('===')

        for job in running_jobs:
            completed = get_job_status(job.job_number)

            if completed:
                new_completed_jobs.append(job)

        for job in new_completed_jobs:
            print('Completed job', job)
            prev_length = len(running_jobs)
            running_jobs.remove(job)
            assert len(running_jobs) == prev_length - 1
            completed_jobs.append(job)

        if len(new_completed_jobs) > 0:
            update_tracker()

    while True:
        check_jobs()

        if len(ready_jobs) == 0 and len(running_jobs) == 0:
            print('All jobs finished, quitting')
            break

        if len(ready_jobs) > 0 and len(running_jobs) < job_cap:
            for _ in range(min(job_cap - len(running_jobs), len(ready_jobs))):
                job = next_job(ready_jobs)
                job_number = run_job(job)
                job.job_number = job_number

                print('Starting job', job)

                ready_jobs.remove(job)
                running_jobs.append(job)
            update_tracker()
        
        time.sleep(tick)

if __name__ == '__main__':
    main()