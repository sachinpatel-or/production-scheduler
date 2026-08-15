"""
Interactive Streamlit dashboard for the Production Scheduler.

Run with:
    streamlit run app/dashboard.py

Provides interactive controls to:
- Run CP-SAT and SPT heuristic on configurable instances
- View Gantt chart and machine utilization
- Compare CP-SAT vs SPT
- Explore the Pareto frontier
- View solver tuning and sensitivity analysis charts

Heavy computations (Pareto frontier, tuning, sensitivity) are lazy:
they only run when the user clicks "Compute", and results are cached in
session_state so they are NOT re-run on every Streamlit rerun.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.instance_generator import generate_jssp_instance
from src.cpsat_solver import solve_jssp_cpsat
from src.heuristic import solve_spt_heuristic
from src.pareto_frontier import compute_pareto_frontier
from src.solver_tuning import run_tuning_experiments, run_sensitivity_analysis


st.set_page_config(page_title="Production Scheduler Dashboard",
                   layout="wide")
st.title("🏭 Production Scheduler — JSSP Dashboard")


# ─────────────────────────────────────────────
# Sidebar configuration
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
num_jobs = st.sidebar.slider("Number of Jobs", 5, 20, 10, step=1)
num_machines = st.sidebar.slider("Number of Machines", 3, 8, 5, step=1)
time_limit = st.sidebar.slider("CP-SAT Time Limit (s)", 5, 60, 30, step=5)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_schedule_dataframe(schedule):
    """Convert a Schedule's assignments into a pandas DataFrame."""
    rows = []
    for a in schedule.assignments:
        rows.append({
            "Job": a.job_id,
            "Op": a.op_index,
            "Machine": f"M{a.machine_id}",
            "Start": a.start_time,
            "End": a.end_time,
            "Processing": a.processing_time,
            "Setup": a.setup_time,
        })
    return pd.DataFrame(rows)


def plotly_gantt(df):
    """Interactive Gantt chart using plotly timeline."""
    df = df.copy()
    df["Job Label"] = df.apply(lambda r: f"J{r['Job']} O{r['Op']}", axis=1)
    fig = px.timeline(df, x_start="Start", x_end="End", y="Machine",
                      color="Job",
                      color_discrete_sequence=px.colors.qualitative.Set3,
                      hover_name="Job Label")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Gantt Chart — Schedule",
                      xaxis_title="Time", yaxis_title="Machine",
                      margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return fig


# Solve base instances (cached — this is fast enough and safe to cache,
# since we only return the two schedules and their data).
@st.cache_data
def solve_base(num_jobs, num_machines, time_limit):
    jobs, machines = generate_jssp_instance(num_jobs, num_machines, seed=42)
    spt = solve_spt_heuristic(jobs, machines)
    cpsat = solve_jssp_cpsat(jobs, machines, time_limit=time_limit, num_workers=4)
    # Return only picklable primitives: makespan, solve_time, status,
    # and a compact assignment list.
    def schedule_to_rows(s):
        return [
            (a.job_id, a.op_index, a.machine_id, a.start_time, a.end_time,
             a.processing_time, a.setup_time)
            for a in s.assignments
        ] + [
            # sentinel line carrying summary
            ("__summary__", s.makespan, round(s.solve_time, 4), s.status,
             tuple(s.machine_utilization.items()))
        ]
    return schedule_to_rows(spt), schedule_to_rows(cpsat)


def rows_to_schedule(rows):
    """Rebuild a lightweight schedule-like object from pickled rows."""
    from src.data_types import Assignment
    assignments = []
    summary = None
    for r in rows:
        if r[0] == "__summary__":
            summary = r
            continue
        job_id, op_index, machine_id, start, end, proc, setup = r
        assignments.append(Assignment(
            job_id=job_id, op_index=op_index, machine_id=machine_id,
            start_time=start, end_time=end,
            processing_time=proc, setup_time=setup))
    makespan, solve_time, status, util_tuple = summary[1], summary[2], summary[3], summary[4]
    class LightSchedule:
        pass
    s = LightSchedule()
    s.assignments = assignments
    s.makespan = makespan
    s.solve_time = solve_time
    s.status = status
    s.machine_utilization = dict(util_tuple)
    s.algorithm = ""
    return s


# ─────────────────────────────────────────────
# Solve base instances
# ─────────────────────────────────────────────
with st.spinner("Solving base instances..."):
    spt_rows, cpsat_rows = solve_base(num_jobs, num_machines, time_limit)
spt = rows_to_schedule(spt_rows)
cpsat = rows_to_schedule(cpsat_rows)

# ─────────────────────────────────────────────
# Top metrics
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("CP-SAT Makespan", cpsat.makespan,
          delta=f"{((spt.makespan - cpsat.makespan) / spt.makespan) * 100:.1f}% vs SPT")
c2.metric("SPT Makespan", spt.makespan)
c3.metric("CP-SAT Solve Time (s)", f"{cpsat.solve_time:.2f}")
c4.metric("CP-SAT Status", cpsat.status)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Schedule", "⚖️ Comparison", "🎯 Pareto Frontier",
     "🔧 Tuning", "📈 Sensitivity"])

# ---- Tab 1: Schedule ----
with tab1:
    algo = st.radio("Algorithm", ["CP-SAT", "SPT Heuristic"], horizontal=True)
    schedule = cpsat if algo == "CP-SAT" else spt
    df = build_schedule_dataframe(schedule)

    gcol, ucol = st.columns([2, 1])
    with gcol:
        st.plotly_chart(plotly_gantt(df), width='stretch')
    with ucol:
        util_df = pd.DataFrame({
            "Machine": [f"M{m}" for m in sorted(schedule.machine_utilization.keys())],
            "Utilization (%)": [schedule.machine_utilization[m] * 100
                                for m in sorted(schedule.machine_utilization.keys())],
        })
        fig = px.bar(util_df, x="Machine", y="Utilization (%)",
                     color="Utilization (%)", color_continuous_scale="Blues",
                     text="Utilization (%)")
        fig.update_layout(title="Machine Utilization", showlegend=False,
                  margin={"l": 0, "r": 0, "t": 40, "b": 0})
        st.plotly_chart(fig, width='stretch')

    with st.expander("View assignment table"):
        st.dataframe(df, width='stretch')

# ---- Tab 2: Comparison ----
with tab2:
    c_left, c_right = st.columns(2)
    comp_df = pd.DataFrame({
        "Algorithm": ["SPT", "CP-SAT"],
        "Makespan": [spt.makespan, cpsat.makespan],
        "Solve Time (s)": [spt.solve_time, cpsat.solve_time],
    })
    with c_left:
        fig = px.bar(comp_df, x="Algorithm", y="Makespan", color="Algorithm",
                     text="Makespan",
                     color_discrete_sequence=["#55A868", "#4C72B0"])
        fig.update_layout(title="Makespan Comparison", showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c_right:
        fig = px.bar(comp_df, x="Algorithm", y="Solve Time (s)", color="Algorithm",
                     text=[f"{t:.2f}" for t in comp_df["Solve Time (s)"]],
                     color_discrete_sequence=["#55A868", "#4C72B0"])
        fig.update_layout(title="Solve Time Comparison", showlegend=False)
        st.plotly_chart(fig, width='stretch')

# ---- Tab 3: Pareto Frontier (lazy) ----
with tab3:
    st.subheader("Pareto Frontier — Makespan vs Utilization")
    st.caption("Click 'Compute Pareto Frontier' to run CP-SAT with multiple "
               "weight combinations. This is slow, so it only runs on demand.")
    key = f"pareto_{num_jobs}_{num_machines}"
    if st.button("Compute Pareto Frontier", key="pareto_btn"):
        with st.spinner("Computing Pareto frontier (runs CP-SAT multiple times)..."):
            st.session_state[key] = compute_pareto_frontier(
                num_jobs=num_jobs, num_machines=num_machines)
    frontier = st.session_state.get(key)
    if frontier:
        p_df = pd.DataFrame({
            "Makespan": [p.makespan for p in frontier],
            "Utilization (%)": [p.utilization * 100 for p in frontier],
            "Weights": [str(p.weights) for p in frontier],
        })
        fig = px.scatter(p_df, x="Makespan", y="Utilization (%)", color="Weights",
                         text="Weights", size=[30] * len(p_df))
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Pareto Frontier")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(p_df, width='stretch')
    else:
        st.info("Pareto frontier not computed yet.")

# ---- Tab 4: Tuning (lazy) ----
with tab4:
    st.subheader("Solver Tuning Results")
    st.caption("Runs CP-SAT with varying time limits and worker counts. "
               "Slow — runs only on demand.")
    key = f"tuning_{num_jobs}_{num_machines}"
    if st.button("Run Tuning Experiments", key="tuning_btn"):
        with st.spinner("Running tuning experiments..."):
            st.session_state[key] = run_tuning_experiments(
                num_jobs=num_jobs, num_machines=num_machines)
    tuning = st.session_state.get(key)
    if tuning:
        t_df = pd.DataFrame([{
            "Param": r.param_name,
            "Value": r.param_value,
            "Makespan": r.makespan,
            "Solve Time (s)": r.solve_time,
            "Status": r.status,
        } for r in tuning])
        fig = px.line(t_df, x="Value", y="Makespan", color="Param",
                      markers=True, title="Makespan vs Tuning Parameter")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(t_df, width='stretch')
    else:
        st.info("Tuning results not computed yet.")

# ---- Tab 5: Sensitivity (lazy) ----
with tab5:
    st.subheader("Sensitivity Analysis")
    st.caption("Runs CP-SAT at 9 perturbation levels. Slow — runs only on demand.")
    key = f"sens_{num_jobs}_{num_machines}"
    if st.button("Run Sensitivity Analysis", key="sens_btn"):
        with st.spinner("Running sensitivity analysis..."):
            st.session_state[key] = run_sensitivity_analysis(
                num_jobs=num_jobs, num_machines=num_machines, perturbation=0.2)
    sens = st.session_state.get(key)
    if sens:
        s_df = pd.DataFrame({
            "Perturbation (%)": [s[0] for s in sens],
            "Makespan": [s[1] for s in sens],
        })
        fig = px.line(s_df, x="Perturbation (%)", y="Makespan",
                      markers=True, title="Makespan vs Processing Time Perturbation")
        fig.add_vline(x=0, line_dash="dash", line_color="gray",
                      annotation_text="Baseline")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(s_df, width='stretch')
    else:
        st.info("Sensitivity results not computed yet.")

st.sidebar.markdown("---")
st.sidebar.info("Built with CP-SAT (OR-Tools), SPT heuristic, and Plotly/Streamlit.")
