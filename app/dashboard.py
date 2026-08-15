"""
Streamlit dashboard for Multi-Objective Production Scheduling (JSSP).
Run: streamlit run app/dashboard.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.instance_generator import generate_instance
from src.cpsat_solver import solve_jssp
from src.pareto_frontier import compute_pareto
from src.heuristic import solve_spt

st.set_page_config(page_title="Production Scheduler (JSSP)", page_icon="🏭", layout="wide")

st.title("Multi-Objective Production Scheduling (JSSP)")
st.markdown("Job Shop Scheduling using OR-Tools CP-SAT with Pareto frontier for makespan vs. utilization.")

# ─── Sidebar ───
st.sidebar.header("Configuration")

num_jobs = st.sidebar.slider("Jobs", 3, 20, 5)
num_machines = st.sidebar.slider("Machines", 2, 10, 3)
time_limit = st.sidebar.slider("Solver Time Limit (s)", 5, 120, 30)
seed = st.sidebar.number_input("Random Seed", value=42)

# ─── Run Single Solve ───
if st.button("Solve (Makespan Minimization)", type="primary"):
    with st.spinner("Generating instance and solving..."):
        instance = generate_instance(num_jobs, num_machines, seed=seed)
        result = solve_jssp(instance, time_limit=time_limit)

    col1, col2, col3 = st.columns(3)
    col1.metric("Makespan", f"{result['makespan']}")
    col2.metric("Status", result['status'])
    col3.metric("Solve Time", f"{result['solve_time']:.1f}s")

    st.subheader("Schedule (Gantt)")
    schedule_df = pd.DataFrame(result['schedule'])
    st.dataframe(schedule_df, use_container_width=True)

    # Gantt chart
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.Set3.colors
    for i, row in schedule_df.iterrows():
        ax.barh(row['machine'], row['end'] - row['start'], left=row['start'],
                color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        ax.text(row['start'] + (row['end'] - row['start']) / 2, row['machine'],
                f"J{row['job']}", ha='center', va='center', fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Production Schedule Gantt Chart")
    st.pyplot(fig)

# ─── Run Pareto Frontier ───
if st.button("Compute Pareto Frontier", type="secondary"):
    with st.spinner("Computing Pareto frontier (makespan vs. utilization)..."):
        instance = generate_instance(num_jobs, num_machines, seed=seed)
        pareto_points = compute_pareto(instance, time_limit=time_limit)

    st.subheader("Pareto Frontier")
    pareto_df = pd.DataFrame(pareto_points)
    st.dataframe(pareto_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pareto_df['makespan'], pareto_df['utilization'], c='#e74c3c', s=80, zorder=5)
    ax.plot(pareto_df['makespan'], pareto_df['utilization'], '--', color='#3498db', alpha=0.5)
    ax.set_xlabel("Makespan (lower is better)")
    ax.set_ylabel("Machine Utilization % (higher is better)")
    ax.set_title("Pareto Frontier: Makespan vs. Utilization")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info("The Pareto frontier shows the trade-off between finishing fast (low makespan) "
            "and keeping machines busy (high utilization). Each point is a non-dominated solution.")
