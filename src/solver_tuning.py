"""
Solver tuning experiments for CP-SAT.

Demonstrates how changing solver parameters affects solution quality and
solve time — a skill explicitly requested in OR job postings.
"""
import time
from src.data_types import TuningResult
from src.cpsat_solver import solve_jssp_cpsat
from src.instance_generator import generate_jssp_instance


def run_tuning_experiments(num_jobs=10, num_machines=5):
    """
    Run solver tuning experiments varying:
    - Time limit (5s, 10s, 30s, 60s)
    - Number of workers (1, 2, 4, 8)
    - Log search progress (on/off)

    Returns list of TuningResult objects.
    """
    jobs, machines = generate_jssp_instance(num_jobs, num_machines, seed=42)
    results = []

    # Experiment 1: Vary time limit
    print("=== Tuning: Time Limit ===")
    for tl in [5, 10, 30, 60]:
        sol = solve_jssp_cpsat(jobs, machines, time_limit=tl, num_workers=4)
        r = TuningResult(
            param_name="time_limit",
            param_value=tl,
            makespan=sol.makespan,
            solve_time=sol.solve_time,
            status=sol.status,
        )
        results.append(r)
        print(f"  time_limit={tl}s: makespan={sol.makespan}, "
              f"status={sol.status}, time={sol.solve_time:.2f}s")

    # Experiment 2: Vary number of workers
    print("\n=== Tuning: Num Workers ===")
    for nw in [1, 2, 4, 8]:
        sol = solve_jssp_cpsat(jobs, machines, time_limit=30, num_workers=nw)
        r = TuningResult(
            param_name="num_workers",
            param_value=nw,
            makespan=sol.makespan,
            solve_time=sol.solve_time,
            status=sol.status,
        )
        results.append(r)
        print(f"  workers={nw}: makespan={sol.makespan}, "
              f"status={sol.status}, time={sol.solve_time:.2f}s")

    return results


def run_sensitivity_analysis(num_jobs=10, num_machines=5, perturbation=0.2):
    """
    Sensitivity analysis: how does makespan change when processing times
    vary by +/- perturbation?

    Returns list of (perturbation_pct, makespan) tuples.
    """
    import numpy as np
    from src.data_types import Operation

    print(f"\n=== Sensitivity Analysis (±{perturbation:.0%}) ===")
    results = []

    for delta in np.linspace(-perturbation, perturbation, 9):
        jobs, machines = generate_jssp_instance(num_jobs, num_machines, seed=42)
        # Perturb processing times
        rng = np.random.RandomState(99)
        for job in jobs:
            for op in job.operations:
                factor = 1.0 + delta * rng.uniform(-1, 1)
                op.processing_time = max(1, int(op.processing_time * factor))

        sol = solve_jssp_cpsat(jobs, machines, time_limit=30, num_workers=4)
        pct = delta * 100
        results.append((pct, sol.makespan))
        print(f"  perturbation={pct:+.1f}%: makespan={sol.makespan}")

    return results
