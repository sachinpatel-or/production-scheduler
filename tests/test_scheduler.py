"""
Tests for the production scheduler.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.instance_generator import generate_jssp_instance
from src.cpsat_solver import solve_jssp_cpsat
from src.heuristic import solve_spt_heuristic


def test_instance_generation():
    jobs, machines = generate_jssp_instance(5, 3, seed=42)
    assert len(jobs) == 5
    assert len(machines) == 3
    for job in jobs:
        assert len(job.operations) == 3  # one per machine


def test_cpsat_solves():
    jobs, machines = generate_jssp_instance(5, 3, seed=42)
    sched = solve_jssp_cpsat(jobs, machines, time_limit=10)
    assert sched.status in ("OPTIMAL", "FEASIBLE")
    assert sched.makespan > 0
    assert len(sched.assignments) == 15  # 5 jobs * 3 ops


def test_spt_heuristic_solves():
    jobs, machines = generate_jssp_instance(5, 3, seed=42)
    sched = solve_spt_heuristic(jobs, machines)
    assert sched.makespan > 0
    assert len(sched.assignments) == 15


def test_cpsat_beats_heuristic():
    """CP-SAT should find a makespan no worse than SPT."""
    jobs, machines = generate_jssp_instance(5, 3, seed=42)
    spt = solve_spt_heuristic(jobs, machines)
    cpsat = solve_jssp_cpsat(jobs, machines, time_limit=10)
    assert cpsat.makespan <= spt.makespan
