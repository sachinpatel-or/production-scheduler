"""
Multi-objective optimization: Pareto frontier for makespan vs utilization.

Generates the trade-off curve between minimizing makespan (total completion time)
and maximizing machine utilization. Each point on the Pareto frontier represents
a solution where you cannot improve one objective without worsening the other.
"""
from src.cpsat_solver import solve_jssp_cpsat
from src.data_types import ParetoPoint


def compute_pareto_frontier(jobs: list, machines: list,
                             weight_pairs=None) -> list:
    """
    Compute the Pareto frontier for makespan vs utilization.

    We solve the multi-objective problem with different weight combinations
    to trace out the trade-off curve.

    Args:
        weight_pairs: List of (makespan_weight, utilization_weight) tuples.
                     Default uses a range of weights.

    Returns:
        List of ParetoPoint objects on the frontier.
    """
    if weight_pairs is None:
        weight_pairs = [
            (1, 0),    # pure makespan
            (1, 1),
            (1, 5),
            (1, 10),
            (1, 20),
            (1, 50),
            (1, 100),  # heavy utilization emphasis
        ]

    points = []

    print("=== Pareto Frontier ===")
    for mw, uw in weight_pairs:
        sol = solve_jssp_cpsat(jobs, machines, time_limit=30,
                               num_workers=4,
                               makespan_weight=mw,
                               utilization_weight=uw)

        avg_util = sum(sol.machine_utilization.values()) / len(machines) if machines else 0
        pt = ParetoPoint(
            makespan=sol.makespan,
            utilization=avg_util,
            weights=(mw, uw),
        )
        points.append(pt)
        print(f"  weights=({mw},{uw}): makespan={sol.makespan}, util={avg_util:.1%}")

    # Filter to non-dominated points
    frontier = []
    for p in points:
        dominated = any(
            other.makespan <= p.makespan and other.utilization >= p.utilization
            for other in points if other is not p
        )
        if not dominated:
            frontier.append(p)

    return frontier
