"""
Constraint Programming solver for JSSP using OR-Tools CP-SAT.

Formulates the job shop scheduling problem as a constraint satisfaction problem:
- Each operation has a start time variable
- Precedence constraints: operations within a job execute in order
- No-overlap constraints: a machine processes one operation at a time
- Objective: minimize makespan (total completion time)
"""
import time
from ortools.sat.python import cp_model
from src.data_types import Job, Machine, Schedule, Assignment


def solve_jssp_cpsat(jobs: list, machines: list,
                     time_limit: float = 30.0,
                     num_workers: int = 4,
                     makespan_weight: int = 1,
                     utilization_weight: int = 0) -> Schedule:
    """
    Solve JSSP using CP-SAT.

    Args:
        jobs: List of Job objects
        machines: List of Machine objects
        time_limit: Solver time limit in seconds
        num_workers: Number of parallel workers
        makespan_weight: Weight for makespan objective
        utilization_weight: Weight for utilization objective (>0 enables multi-objective)

    Returns:
        Schedule with assignments and metrics
    """
    start_time = time.time()

    model = cp_model.CpModel()

    # Compute horizon (upper bound on makespan)
    horizon = sum(op.processing_time + op.setup_time
                  for job in jobs for op in job.operations)

    # ─── Variables ───
    # start_var[op_key] = start time of operation
    start_vars = {}
    end_vars = {}
    intervals = {}
    op_map = {}  # (job_id, op_index) -> Operation

    for job in jobs:
        for op in job.operations:
            key = (job.id, op.op_index)
            suffix = f"j{job.id}_o{op.op_index}"

            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            duration = op.processing_time + op.setup_time
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")

            start_vars[key] = start
            end_vars[key] = end
            intervals[key] = interval
            op_map[key] = op

    # ─── Constraints ───

    # 1. Precedence: operations within a job execute in order
    for job in jobs:
        for i in range(len(job.operations) - 1):
            key1 = (job.id, i)
            key2 = (job.id, i + 1)
            model.Add(start_vars[key2] >= end_vars[key1])

    # 2. No-overlap: each machine processes one operation at a time
    machine_intervals = {m.id: [] for m in machines}
    for key, interval in intervals.items():
        op = op_map[key]
        machine_intervals[op.machine_id].append(interval)

    for machine_id, machine_intervals_list in machine_intervals.items():
        if len(machine_intervals_list) > 1:
            model.AddNoOverlap(machine_intervals_list)

    # ─── Objective ───
    # Makespan = max of all end times
    makespan = model.NewIntVar(0, horizon, "makespan")
    all_ends = [end_vars[key] for key in end_vars]
    model.AddMaxEquality(makespan, all_ends)

    if utilization_weight > 0:
        # Multi-objective: minimize makespan, maximize utilization
        # Utilization = total processing time / (num_machines * makespan)
        # We maximize utilization by minimizing (num_machines * makespan - total_processing_time)
        total_processing = sum(op.processing_time + op.setup_time
                               for job in jobs for op in job.operations)
        idle_time = model.NewIntVar(0, horizon * len(machines), "idle_time")
        model.Add(idle_time == len(machines) * makespan - total_processing)

        model.Minimize(makespan_weight * makespan + utilization_weight * idle_time)
    else:
        model.Minimize(makespan_weight * makespan)

    # ─── Solve ───
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    solve_time = time.time() - start_time

    # ─── Extract solution ───
    assignments = []
    status_str = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for key, start_var in start_vars.items():
            op = op_map[key]
            start_val = solver.Value(start_var)
            end_val = solver.Value(end_vars[key])
            assignments.append(Assignment(
                job_id=key[0],
                op_index=key[1],
                machine_id=op.machine_id,
                start_time=start_val,
                end_time=end_val,
                processing_time=op.processing_time,
                setup_time=op.setup_time,
            ))

        makespan_val = solver.Value(makespan)

        # Compute machine utilization
        util = {}
        for m in machines:
            busy = sum(a.processing_time + a.setup_time
                       for a in assignments if a.machine_id == m.id)
            util[m.id] = busy / makespan_val if makespan_val > 0 else 0

        return Schedule(
            assignments=assignments,
            makespan=makespan_val,
            machine_utilization=util,
            solve_time=solve_time,
            algorithm="CP-SAT",
            status=status_str,
        )

    return Schedule(
        assignments=[],
        makespan=0,
        solve_time=solve_time,
        algorithm="CP-SAT",
        status=status_str,
    )
