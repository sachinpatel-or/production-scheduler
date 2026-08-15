"""
Benchmark runner: compare CP-SAT vs SPT heuristic across instance sizes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.instance_generator import generate_jssp_instance
from src.cpsat_solver import solve_jssp_cpsat
from src.heuristic import solve_spt_heuristic
from src.solver_tuning import run_tuning_experiments, run_sensitivity_analysis
from src.pareto_frontier import compute_pareto_frontier


def main():
    print("=" * 70)
    print("JSSP Benchmark: CP-SAT vs SPT Heuristic")
    print("=" * 70)

    for num_jobs, num_machines in [(5, 3), (10, 5), (15, 5)]:
        jobs, machines = generate_jssp_instance(num_jobs, num_machines, seed=42)

        spt = solve_spt_heuristic(jobs, machines)
        cpsat = solve_jssp_cpsat(jobs, machines, time_limit=30, num_workers=4)

        improvement = ((spt.makespan - cpsat.makespan) / spt.makespan) * 100
        print(f"\n{num_jobs} jobs, {num_machines} machines:")
        print(f"  SPT:       makespan={spt.makespan}, time={spt.solve_time:.4f}s")
        print(f"  CP-SAT:    makespan={cpsat.makespan}, time={cpsat.solve_time:.2f}s, status={cpsat.status}")
        print(f"  Improvement: {improvement:.1f}%")

    # Run tuning experiments
    run_tuning_experiments(num_jobs=10, num_machines=5)

    # Run sensitivity analysis
    run_sensitivity_analysis(num_jobs=10, num_machines=5, perturbation=0.2)

    # Compute Pareto frontier
    frontier = compute_pareto_frontier(num_jobs=10, num_machines=5)
    print(f"\nPareto frontier: {len(frontier)} non-dominated points")


if __name__ == "__main__":
    main()
