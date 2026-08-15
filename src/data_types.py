"""
Core data structures for the Production Scheduling project.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Job:
    """A job with multiple operations, each requiring a machine."""
    id: int
    operations: List['Operation']  # ordered: op 0 must complete before op 1 starts


@dataclass
class Operation:
    """A single operation within a job."""
    job_id: int
    op_index: int        # position in the job (0, 1, 2, ...)
    machine_id: int      # which machine this operation runs on
    processing_time: int # time units to complete
    setup_time: int = 0  # setup time before this operation


@dataclass
class Machine:
    """A machine that can process one operation at a time."""
    id: int
    name: str = ""
    available_from: int = 0  # earliest time machine is available


@dataclass
class Schedule:
    """A complete production schedule."""
    assignments: List['Assignment'] = field(default_factory=list)
    makespan: int = 0
    machine_utilization: Dict[int, float] = field(default_factory=dict)
    solve_time: float = 0.0
    algorithm: str = ""
    status: str = ""  # OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN

    @property
    def num_jobs_completed(self) -> int:
        return len(set(a.job_id for a in self.assignments))


@dataclass
class Assignment:
    """An operation assigned to a machine at a specific time."""
    job_id: int
    op_index: int
    machine_id: int
    start_time: int
    end_time: int
    processing_time: int
    setup_time: int = 0


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    makespan: int
    utilization: float
    weights: tuple  # (makespan_weight, utilization_weight)


@dataclass
class TuningResult:
    """Result of a solver tuning experiment."""
    param_name: str
    param_value: object
    makespan: int
    solve_time: float
    status: str
