"""
Generate realistic Job Shop Scheduling Problem (JSSP) instances.
"""
import numpy as np
from src.data_types import Job, Operation, Machine


def generate_jssp_instance(num_jobs: int = 10, num_machines: int = 5,
                           min_time: int = 5, max_time: int = 50,
                           seed: int = 42) -> tuple:
    """
    Generate a synthetic JSSP instance.

    Each job has `num_machines` operations, one per machine, in random order.
    Processing times are drawn uniformly from [min_time, max_time].
    Setup times are 0-10% of processing time.

    Returns:
        (jobs: List[Job], machines: List[Machine])
    """
    rng = np.random.RandomState(seed)
    machines = [Machine(id=m, name=f"M{m}") for m in range(num_machines)]
    jobs = []

    for j in range(num_jobs):
        machine_order = rng.permutation(num_machines)
        operations = []
        for op_idx, machine_id in enumerate(machine_order):
            proc_time = int(rng.uniform(min_time, max_time))
            setup = int(rng.uniform(0, proc_time * 0.1))
            operations.append(Operation(
                job_id=j, op_index=op_idx,
                machine_id=int(machine_id),
                processing_time=proc_time,
                setup_time=setup,
            ))
        jobs.append(Job(id=j, operations=operations))

    return jobs, machines
