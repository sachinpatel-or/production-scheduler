"""
Dispatching heuristics for JSSP — used as baselines for comparison with CP-SAT.

Shortest Processing Time (SPT) first: assign operations with shortest
processing time to machines greedily. Fast but suboptimal.
"""
import time
from src.data_types import Job, Machine, Schedule, Assignment


def solve_spt_heuristic(jobs: list, machines: list) -> Schedule:
    """
    Shortest Processing Time dispatching heuristic.

    Strategy: At each step, assign the available operation with the shortest
    processing time to its machine. This minimizes average waiting time but
    typically produces 20-40% worse makespan than optimal.
    """
    start_time = time.time()

    # Track machine available times
    machine_available = {m.id: m.available_from for m in machines}
    # Track job progress (next operation index)
    job_next_op = {j.id: 0 for j in jobs}
    # Track job completion time
    job_end_time = {j.id: 0 for j in jobs}

    assignments = []
    total_ops = sum(len(j.operations) for j in jobs)
    completed = 0

    while completed < total_ops:
        # Collect all available operations
        available = []
        for job in jobs:
            idx = job_next_op[job.id]
            if idx < len(job.operations):
                op = job.operations[idx]
                earliest_start = max(machine_available[op.machine_id], job_end_time[job.id])
                available.append((op.processing_time, earliest_start, job.id, op))

        if not available:
            break

        # Sort by processing time (shortest first), then earliest start
        available.sort(key=lambda x: (x[0], x[1]))

        # Assign the first one
        proc_time, earliest_start, job_id, op = available[0]
        start = earliest_start
        end = start + op.processing_time + op.setup_time

        assignments.append(Assignment(
            job_id=job_id, op_index=op.op_index,
            machine_id=op.machine_id,
            start_time=start, end_time=end,
            processing_time=op.processing_time,
            setup_time=op.setup_time,
        ))

        machine_available[op.machine_id] = end
        job_end_time[job_id] = end
        job_next_op[job_id] += 1
        completed += 1

    makespan = max(a.end_time for a in assignments) if assignments else 0

    # Compute utilization
    util = {}
    for m in machines:
        busy = sum(a.processing_time + a.setup_time
                   for a in assignments if a.machine_id == m.id)
        util[m.id] = busy / makespan if makespan > 0 else 0

    return Schedule(
        assignments=assignments,
        makespan=makespan,
        machine_utilization=util,
        solve_time=time.time() - start_time,
        algorithm="SPT Heuristic",
        status="FEASIBLE",
    )
