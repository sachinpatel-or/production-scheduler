# Production Scheduler — Multi-Objective JSSP with CP-SAT

Solves the Job Shop Scheduling Problem (JSSP) using constraint programming,
with multi-objective optimization, solver tuning, and sensitivity analysis.

## Features

- **CP-SAT solver** (OR-Tools) for exact constraint programming
- **SPT dispatching heuristic** as baseline comparison
- **Multi-objective optimization**: Pareto frontier for makespan vs. utilization
- **Solver tuning experiments**: vary time limit, workers, parameters
- **Sensitivity analysis**: perturb processing times ±20% and measure impact

## Quick Start

```bash
pip install -r requirements.txt
python -m benchmarks.run_comparison
python -m pytest tests/ -v
```

## Skills Demonstrated

- Constraint programming (CP-SAT)
- Multi-objective optimization and Pareto frontiers
- Solver parameter tuning and performance analysis
- Sensitivity analysis under uncertainty
- Dispatching heuristics as baselines
