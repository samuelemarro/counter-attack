def get_job_id(domain, architecture, test_name, index):
    return f'{domain}-{architecture}-{test_name}-{index}'

class Job:
    def __init__(self, domain, architecture, test_name, index, priority=None):
        self.domain = domain
        self.architecture = architecture
        self.test_name = test_name
        self.index = index
        self.job_number = None
        self.status = None
        self.priority = priority
    
    def __str__(self) -> str:
        return f'{self.domain} {self.architecture} {self.test_name} {self.index} (job number: {self.job_number}, status: {self.status}, priority: {self.priority})'
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, o: object) -> bool:
        # Ignores job number, status and priority
        if isinstance(o, Job):
            return self.test_name == self.test_name and self.domain == o.domain and \
            self.architecture == o.architecture and self.index == o.index
        return False

    def __hash__(self) -> int:
        return hash(self.index)

    @property
    def unique_id(self):
        return get_job_id(self.domain, self.architecture, self.test_name, self.index)
    
    def to_csv(self):
        return f'{self.domain},{self.architecture},{self.test_name},{self.index}'

def read_jobs(tracker_path):
    with open(tracker_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    jobs = {}

    for line in lines:
        split_line = [subelement.strip() for subelement in line.split(',')]
        if len(split_line) <= 1:
            continue

        if split_line[0] == 'QUEUED':
            # Format: QUEUED, cifar10, c, relu, 0
            job = Job(split_line[1], split_line[2], split_line[3], int(split_line[4]))
            job.status = 'QUEUED'
            jobs[job.unique_id] = job
        elif split_line[0] == 'REGISTERED':
            # Format: REGISTERED, cifar10, c, relu, 0, 390922
            job_number = int(split_line[5])
            job_id = get_job_id(split_line[1], split_line[2], split_line[3], int(split_line[4]))
            jobs[job_id].job_number = job_number
        elif split_line[0] == 'STARTED':
            # Format: STARTED, cifar10, c, relu, 0
            job_id = get_job_id(split_line[1], split_line[2], split_line[3], int(split_line[4]))
            jobs[job_id].status = 'RUNNING'
        elif split_line[0] == 'FINISHED':
            # Format: FINISHED, cifar10, c, relu, 0
            job_id = get_job_id(split_line[1], split_line[2], split_line[3], int(split_line[4]))
            jobs[job_id].status = 'FINISHED'
        else:
            raise NotImplementedError(f'Unsupported descriptor "{split_line[0]}".')

    return jobs


def remove_jobs(tracker_path, jobs_to_remove, destination_path):
    with open(tracker_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    cleaned_lines = []
    
    for line in lines:
        split_line = [subelement.strip() for subelement in line.split(',')]
        if len(split_line) <= 1:
            continue

        remove = False

        for job in jobs_to_remove:
            if split_line[1] == job.domain and split_line[2] == job.architecture and split_line[3] == job.test_name and int(split_line[4]) == job.index:
                remove = True
                break

        if not remove:
            cleaned_lines.append(line)
    
    with open(destination_path, 'w') as f:
        f.write('\n'.join(cleaned_lines) + '\n')