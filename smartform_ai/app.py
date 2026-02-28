import streamlit as st
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'axes.edgecolor': '#DDDDDD',
    'axes.labelcolor': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'text.color': '#222222',
    'grid.color': '#EEEEEE',
    'grid.linestyle': '--',
    'axes.grid': True,
})

from boq_generator import generate_boq
from repetition_engine import detect_repetitions, plot_repetition_bar_chart
from optimization_engine import optimize_formwork_sets
from inventory_simulator import simulate_timeline, plot_inventory_timeline

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartForm AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
# NOTE: All KPI card colours are set via INLINE style= attributes below so
# Streamlit's dark-theme CSS cannot cascade over them.
st.markdown("""
<style>
/* Force light background everywhere */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
[data-testid="block-container"] {
    background-color: #F0F2F5 !important;
    color: #1A1A2E !important;
}

/* Sidebar — light theme, left border accent */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] * {
    color: #1A1A2E !important;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #374151 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1A1A2E !important;
}

/* Page headers */
h1, h2, h3, h4, h5, h6 {
    color: #1A1A2E !important;
    font-family: 'Segoe UI', sans-serif !important;
}

/* Dividers */
hr { border-color: #D1D5DB !important; }

/* Section label styling */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 4px;
}

/* Dataframe text always dark */
[data-testid="stDataFrame"] * { color: #1A1A2E !important; }

/* Remove default streamlit padding from metric widgets */
[data-testid="stMetric"] { background: transparent; }

/* Spinner text */
.stSpinner > div > div { color: #1A1A2E !important; }

/* Info box */
[data-testid="stInfoAlert"] { color: #1A1A2E !important; }
</style>
""", unsafe_allow_html=True)


# ── KPI HTML Helper (100% inline — Streamlit cannot override) ──────────────────
def kpi_card(title: str, value: str, sub: str, accent: str = "#1565C0", sub_color: str = "#2E7D32"):
    return f"""
    <div style="
        background: #FFFFFF;
        border-radius: 10px;
        padding: 22px 20px 18px 20px;
        border-left: 5px solid {accent};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 6px;
        min-height: 115px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #6B7280;
            margin-bottom: 8px;
        ">{title}</div>
        <div style="
            font-size: 26px;
            font-weight: 800;
            color: #1A1A2E;
            line-height: 1.1;
            margin-bottom: 6px;
        ">{value}</div>
        <div style="
            font-size: 13px;
            font-weight: 600;
            color: {sub_color};
        ">{sub}</div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #E5E7EB; margin-bottom:16px;">
        <span style="font-size:26px;">🏗️</span>
        <span style="font-size:19px; font-weight:800; color:#1A1A2E; margin-left:8px;">SmartForm AI</span>
        <div style="font-size:11px; color:#6B7280; margin-top:4px; letter-spacing:0.5px;">
            Formwork Optimization Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    reuse_cycle = st.slider("Reuse Cycle (Days)", min_value=3, max_value=21, value=7)
    carrying_cost_rate = st.slider("Carrying Cost %", min_value=5, max_value=30, value=15) / 100.0
    st.markdown("---")
    st.markdown("**📂 Upload Data**")
    uploaded_elements = st.file_uploader("structural_elements.csv", type=['csv'])
    uploaded_schedule  = st.file_uploader("schedule.csv (optional)", type=['csv'])
    st.markdown("---")
    st.caption("Demo data is auto-loaded if no file is uploaded.")


# ── Data Loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    if f is not None:
        return pd.read_csv(f)
    if os.path.exists("structural_elements.csv"):
        return pd.read_csv("structural_elements.csv")
    return None


# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    padding: 20px 0 6px 0;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 28px;
">
    <div style="font-size:11px; font-weight:700; letter-spacing:1.6px;
                text-transform:uppercase; color:#6B7280; margin-bottom:4px;">
        FORMWORK OPTIMIZATION PLATFORM
    </div>
    <h1 style="margin:0; font-size:28px; font-weight:800; color:#1A1A2E; line-height:1.2;">
        SmartForm AI Dashboard
    </h1>
    <p style="margin:6px 0 0 0; font-size:14px; color:#6B7280;">
        Structural repetition detection · Procurement optimization · Inventory simulation
    </p>
</div>
""", unsafe_allow_html=True)


df_elements = load_data(uploaded_elements)

if df_elements is None:
    st.info("Upload **structural_elements.csv** using the sidebar, or run `python generate_mock_data.py` to generate demo data.")
    st.stop()

# ── Processing Pipeline ───────────────────────────────────────────────────────
df_elements = generate_boq(df_elements)
df_clustered, df_summary = detect_repetitions(df_elements)

with st.spinner("Running PuLP optimization…"):
    df_opt = optimize_formwork_sets(df_clustered, reuse_cycle_days=reuse_cycle)

# KPI calculations
opt_cost   = df_opt['Optimized_Procurement_Cost'].sum()
naive_cost = df_opt['Naive_Procurement_Cost'].sum()
proc_savings = naive_cost - opt_cost
carry_savings = (naive_cost - opt_cost) * carrying_cost_rate
total_savings = proc_savings + carry_savings

proc_red_pct  = (proc_savings / naive_cost * 100) if naive_cost else 0
total_opt_sets   = df_opt['Required_Sets'].sum()
total_naive_sets = df_opt['Naive_Sets'].sum()
excess_red_pct   = ((total_naive_sets - total_opt_sets) / total_naive_sets * 100) if total_naive_sets else 0
carry_red_pct    = proc_red_pct  # same ratio


# ── Section 1 – KPI Cards ─────────────────────────────────────────────────────
st.markdown(
    "<div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
    "color:#6B7280;margin-bottom:12px;'>Executive Summary</div>",
    unsafe_allow_html=True
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi_card(
        "Optimised Procurement Cost",
        f"₹{opt_cost:,.0f}",
        f"↓ {proc_red_pct:.1f}% vs no-reuse baseline",
        accent="#1565C0"
    ), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card(
        "Total Projected Savings",
        f"₹{total_savings:,.0f}",
        "Capex + carrying costs combined",
        accent="#00695C"
    ), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card(
        "Optimised Inventory",
        f"{total_opt_sets} sets",
        f"↓ {excess_red_pct:.1f}% excess vs naive",
        accent="#E65100"
    ), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card(
        "Carrying Cost Saved",
        f"₹{carry_savings:,.0f}",
        f"At {carrying_cost_rate*100:.0f}% carrying rate",
        accent="#6A1B9A"
    ), unsafe_allow_html=True)

st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
st.divider()


# ── Section 2 – Repetition Clusters ──────────────────────────────────────────
st.markdown(
    "<div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
    "color:#6B7280;margin-bottom:4px;'>01</div>"
    "<h3 style='margin:0 0 16px 0;color:#1A1A2E;'>Repetition Detection & BoQ Clusters</h3>",
    unsafe_allow_html=True
)

col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    st.markdown("<p style='font-size:13px;color:#6B7280;margin-bottom:8px;'>Top 10 clusters by frequency</p>", unsafe_allow_html=True)
    st.dataframe(df_summary.head(10), use_container_width=True, hide_index=True)
with col_b:
    # Re-plot with refined look
    fig_rep, ax = plt.subplots(figsize=(8, 4.2))
    top10 = df_summary.head(10)
    colours = ["#1565C0"] * len(top10)
    colours[0] = "#0D47A1"
    bars = ax.barh(top10['Cluster_ID'][::-1], top10['Frequency'][::-1], color=colours[::-1],
                   height=0.6, edgecolor='none')
    # Label bars
    for bar, val in zip(bars, top10['Frequency'][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', ha='left', fontsize=9, color='#444')
    ax.set_xlabel("Repetition Count", fontsize=10)
    ax.set_title("Element Repetition by Cluster", fontsize=12, fontweight='bold', color='#1A1A2E', pad=12)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, top10['Frequency'].max() * 1.12)
    plt.tight_layout()
    st.pyplot(fig_rep)

st.divider()


# ── Section 3 – Optimisation Results ─────────────────────────────────────────
st.markdown(
    "<div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
    "color:#6B7280;margin-bottom:4px;'>02</div>"
    "<h3 style='margin:0 0 4px 0;color:#1A1A2E;'>Procurement Optimisation Results</h3>"
    "<p style='font-size:13px;color:#6B7280;margin-bottom:14px;'>PuLP linear programming · "
    f"{reuse_cycle}-day reuse cycle constraint</p>",
    unsafe_allow_html=True
)

display_df = df_opt[['Cluster_ID','Required_Sets','Naive_Sets',
                      'Optimized_Procurement_Cost','Cost_Savings','Cost_Reduction_%']].copy()
display_df.columns = ['Cluster', 'Optimised Sets', 'Naive Sets',
                      'Optimised Cost (₹)', 'Savings (₹)', 'Reduction %']

st.dataframe(
    display_df.style
    .format({'Optimised Cost (₹)': '₹{:,.0f}', 'Savings (₹)': '₹{:,.0f}', 'Reduction %': '{:.1f}%'})
    .background_gradient(subset=['Savings (₹)'], cmap='YlGn')
    .set_properties(**{'font-size': '13px'}),
    use_container_width=True,
    hide_index=True,
    height=320
)

# Summary metrics row
m1, m2, m3 = st.columns(3)
m1.metric("Total Optimised Sets", f"{total_opt_sets:,}", f"-{total_naive_sets - total_opt_sets:,} vs naive")
m2.metric("Total Cost Reduction", f"₹{proc_savings:,.0f}", f"-{proc_red_pct:.1f}%")
m3.metric("Avg Cluster Savings", f"₹{(proc_savings / len(df_opt) if len(df_opt) else 0):,.0f}", "per cluster")

st.divider()


# ── Section 4 – Inventory Timeline ────────────────────────────────────────────
st.markdown(
    "<div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
    "color:#6B7280;margin-bottom:4px;'>03</div>"
    "<h3 style='margin:0 0 4px 0;color:#1A1A2E;'>Inventory Simulation Timeline</h3>",
    unsafe_allow_html=True
)

df_timeline = simulate_timeline(df_clustered, df_opt, reuse_cycle_days=reuse_cycle)
reuse_eff = (
    df_timeline['Required Sets (Active)'].sum() /
    (total_opt_sets * len(df_timeline)) * 100
) if total_opt_sets else 0

st.markdown(
    f"<p style='font-size:13px;color:#6B7280;margin-bottom:14px;'>"
    f"Average reuse efficiency over project lifespan: "
    f"<strong style='color:#1A1A2E;'>{reuse_eff:.1f}%</strong></p>",
    unsafe_allow_html=True
)

# Custom-styled timeline chart
fig_tl, ax2 = plt.subplots(figsize=(12, 4.2))
ax2.fill_between(df_timeline['Date'], df_timeline['Required Sets (Active)'],
                 alpha=0.15, color='#1565C0')
ax2.plot(df_timeline['Date'], df_timeline['Available Sets (Inventory)'],
         label='Inventory Available', color='#B0BEC5', linewidth=1.8,
         linestyle='--', dashes=(5, 3))
ax2.plot(df_timeline['Date'], df_timeline['Required Sets (Active)'],
         label='In-Use (Active)', color='#1565C0', linewidth=2.2)
ax2.set_xlabel("Project Date", fontsize=10)
ax2.set_ylabel("Formwork Sets", fontsize=10)
ax2.set_title("Formwork Usage vs Available Inventory", fontsize=12,
              fontweight='bold', color='#1A1A2E', pad=12)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=10, framealpha=0.9)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.tight_layout()
st.pyplot(fig_tl)

st.divider()


# ── Section 5 – Final Summary Banner ─────────────────────────────────────────
st.markdown(f"""
<div style="
    background: #1A1A2E;
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 8px;
    display: flex;
    align-items: flex-start;
">
    <div style="flex:1; margin-right:40px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                    color:#7986CB;margin-bottom:8px;">Project Final Summary</div>
        <div style="font-size:22px;font-weight:800;color:#FFFFFF;margin-bottom:4px;">
            ₹{total_savings:,.0f} estimated total savings
        </div>
        <div style="font-size:13px;color:#9FA8DA;margin-top:4px;">
            {excess_red_pct:.1f}% excess inventory reduction &nbsp;·&nbsp;
            {reuse_eff:.1f}% reuse efficiency &nbsp;·&nbsp;
            {reuse_cycle}-day cycle
        </div>
    </div>
    <div style="text-align:right; min-width:120px;">
        <div style="font-size:11px;color:#7986CB;margin-bottom:4px;letter-spacing:1px;">OPTIMISED SETS</div>
        <div style="font-size:32px;font-weight:900;color:#FFFFFF;">{total_opt_sets}</div>
        <div style="font-size:11px;color:#5C6BC0;">vs {total_naive_sets} naive</div>
    </div>
</div>
""", unsafe_allow_html=True)
