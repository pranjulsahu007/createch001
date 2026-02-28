import streamlit as st
import pandas as pd
import os
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'axes.edgecolor':   '#DDDDDD',
    'axes.labelcolor':  '#333333',
    'xtick.color':      '#555555',
    'ytick.color':      '#555555',
    'text.color':       '#222222',
    'grid.color':       '#EEEEEE',
    'grid.linestyle':   '--',
    'axes.grid':        True,
})

from boq_generator          import generate_boq
from repetition_engine      import detect_repetitions, plot_repetition_bar_chart
from optimization_engine    import optimize_formwork_sets
from inventory_simulator    import simulate_timeline, plot_inventory_timeline
from procurement_scheduler  import generate_procurement_schedule
from revit_importer         import load_revit_file, auto_map_columns, convert_revit_to_smartform
from dxf_parser             import get_layers, parse_dxf, EZDXF_OK

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartForm AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="block-container"] {
    background-color: #F0F2F5 !important;
    color: #1A1A2E !important;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] * { color: #1A1A2E !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #374151 !important;
    font-size: 13px !important;
}
h1, h2, h3, h4 { color: #1A1A2E !important; }
hr { border-color: #D1D5DB !important; }
[data-testid="stDataFrame"] * { color: #1A1A2E !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: inline KPI card (Streamlit cannot override inline styles) ──────────
def kpi_card(title, value, sub, accent="#1565C0", sub_color="#2E7D32"):
    return f"""
    <div style="background:#FFFFFF;border-radius:10px;padding:20px 18px 16px;
                border-left:5px solid {accent};
                box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:6px;min-height:110px;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1.4px;
                  text-transform:uppercase;color:#6B7280;margin-bottom:8px;">{title}</div>
      <div style="font-size:24px;font-weight:800;color:#1A1A2E;
                  line-height:1.15;margin-bottom:5px;">{value}</div>
      <div style="font-size:12px;font-weight:600;color:{sub_color};">{sub}</div>
    </div>"""

def section_header(num, title, subtitle=""):
    sub_html = f"<p style='font-size:13px;color:#6B7280;margin:2px 0 14px;'>{subtitle}</p>" if subtitle else "<div style='margin-bottom:14px;'></div>"
    return f"""
    <div style='margin-top:8px;'>
      <div style='font-size:10px;font-weight:700;letter-spacing:1.5px;
                  text-transform:uppercase;color:#9CA3AF;margin-bottom:2px;'>{num:02d}</div>
      <h3 style='margin:0;font-size:20px;font-weight:800;color:#1A1A2E;'>{title}</h3>
      {sub_html}
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 14px;border-bottom:1px solid #E5E7EB;margin-bottom:16px;">
      <span style="font-size:24px;">🏗️</span>
      <span style="font-size:18px;font-weight:800;color:#1A1A2E;margin-left:8px;">SmartForm AI</span>
      <div style="font-size:11px;color:#6B7280;margin-top:3px;">Formwork Optimization Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature 3: Advanced Reuse Cycle ───────────────────────────────────────
    st.markdown("**⚙️ Reuse Cycle (Feature 3)**")
    curing_days    = st.slider("Curing Duration (days)",          1, 14, 5)
    stripping_days = st.slider("Stripping Duration (days)",       0,  3, 1)
    handling_days  = st.slider("Cleaning & Handling (days)",      0,  3, 1)
    reuse_cycle    = curing_days + stripping_days + handling_days
    st.markdown(
        f"<div style='background:#EFF6FF;border-radius:6px;padding:8px 12px;margin-bottom:12px;"
        f"font-size:13px;color:#1E40AF;'>"
        f"⏱ Effective reuse cycle: <strong>{reuse_cycle} days</strong>"
        f"<br><span style='font-size:11px;color:#6B7280;'>"
        f"{curing_days}d curing + {stripping_days}d strip + {handling_days}d handling</span></div>",
        unsafe_allow_html=True
    )

    # ── Feature 1: Reuse Life Limit ───────────────────────────────────────────
    st.markdown("**♻️ Reuse Life Limit (Feature 1)**")
    enable_life_limit = st.toggle("Enable reuse life limit", value=False)
    max_reuse_count   = -1
    if enable_life_limit:
        max_reuse_count = st.slider("Max uses per set (before write-off)", 2, 50, 10)
        st.caption(f"Sets are written off after {max_reuse_count} pours.")

    # ── Feature 2: Zone Constraint ────────────────────────────────────────────
    st.markdown("**🗺 Zone Splitting (Feature 2)**")
    enable_zones = st.toggle("Split inventory by zone", value=False)

    # ── Feature 4: Procurement Lead Time ─────────────────────────────────────
    st.markdown("**📦 Procurement Lead Time (Feature 4)**")
    lead_time_days = st.slider("Supplier lead time (days)", 0, 30, 5)

    # ── Carrying cost ─────────────────────────────────────────────────────────
    st.markdown("**💰 Carrying Cost**")
    carrying_cost_rate = st.slider("Carrying Cost %", 5, 30, 15) / 100.0

    st.divider()
    st.markdown("**📂 Upload Data**")
    uploaded_elements = st.file_uploader("structural_elements.csv", type=['csv'])
    st.file_uploader("schedule.csv (optional)", type=['csv'])
    st.divider()
    st.caption("Demo data is auto-loaded if no file is uploaded.")


# ── Data loader ───────────────────────────────────────────────────────────────
def _file_mtime(path):
    """Return file modification time as cache-busting key."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0

@st.cache_data
def load_data(f, _mtime=None):   # _mtime is ignored at runtime but busts the cache
    if f is not None:
        df = pd.read_csv(f)
    elif os.path.exists("structural_elements.csv"):
        df = pd.read_csv("structural_elements.csv")
    else:
        return None
    # Normalise column names (strip stray leading characters)
    df.columns = [c.lstrip('n').strip() for c in df.columns]
    return df


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:18px 0 6px;border-bottom:1px solid #E5E7EB;margin-bottom:24px;">
  <div style="font-size:10px;font-weight:700;letter-spacing:1.6px;
              text-transform:uppercase;color:#6B7280;margin-bottom:4px;">
    FORMWORK OPTIMIZATION PLATFORM
  </div>
  <h1 style="margin:0;font-size:26px;font-weight:800;color:#1A1A2E;line-height:1.2;">
    SmartForm AI Dashboard
  </h1>
  <p style="margin:5px 0 0;font-size:13px;color:#6B7280;">
    Repetition detection · LP optimisation · Inventory simulation · Procurement scheduling
  </p>
</div>
""", unsafe_allow_html=True)



# ══ DATA IMPORT SECTION ══════════════════════════════════════════════════════
st.markdown(
    "<div style='font-size:10px;font-weight:700;letter-spacing:1.5px;"
    "text-transform:uppercase;color:#9CA3AF;margin-bottom:8px;'>Data Import</div>",
    unsafe_allow_html=True
)
tab_csv, tab_revit, tab_dxf = st.tabs([
    "📄 CSV / SmartForm Format",
    "🏗️ Revit Schedule Export",
    "📍 AutoCAD DXF",
])

if 'imported_df' not in st.session_state:
    st.session_state['imported_df'] = None

# ─ Tab 1: Standard CSV ──────────────────────────────────────────────────────
with tab_csv:
    st.markdown(
        "Upload **structural_elements.csv** via the sidebar, or use demo data.  \n"
        "Required columns: `Element_ID` · `Type` · `Length` · `Width` · `Height` (m) "
        "· `Floor` · `Zone` · `Casting_Date` · `Formwork_Cost_per_Set`"
    )
    template = pd.DataFrame([{
        'Element_ID': 'E-001', 'Type': 'Column', 'Length': 0.5, 'Width': 0.5,
        'Height': 3.0, 'Floor': 1, 'Zone': 'Zone-A', 'Casting_Date': '2026-03-01',
        'Formwork_Cost_per_Set': 1200, 'Replacement_Cost_per_Set': 1020, 'Max_Reuse_Count': 10
    }])
    st.download_button("⬇ Download blank template",
                       data=template.to_csv(index=False),
                       file_name="smartform_template.csv", mime="text/csv")
    if st.session_state['imported_df'] is not None:
        if st.button("🗑 Clear imported data — revert to CSV / demo data"):
            st.session_state['imported_df'] = None
            st.rerun()

# ─ Tab 2: Revit ─────────────────────────────────────────────────────────────
with tab_revit:
    st.markdown(
        "💡 **Revit export path:** View → Schedules → Structural Column / Framing / Floor Schedule → "
        "Export → save as **.csv** or **.xlsx**"
    )
    revit_file = st.file_uploader("Upload Revit schedule", type=['csv','xlsx','xls'], key="revit_upload")
    if revit_file:
        try:
            import io as _io2
            file_bytes = revit_file.read()
            revit_raw  = load_revit_file(_io2.BytesIO(file_bytes), revit_file.name)
            st.success(f"Loaded {len(revit_raw)} rows · {len(revit_raw.columns)} columns from Revit export.")
            auto_map = auto_map_columns(revit_raw)
            all_cols = ['(none)'] + list(revit_raw.columns)

            st.markdown("**🔄 Map Revit columns → SmartForm fields** (auto-detected — adjust if wrong)")
            mc = st.columns(4)
            def _sel(lbl, fld, col):
                default = auto_map.get(fld)
                idx = all_cols.index(default) if default in all_cols else 0
                return col.selectbox(lbl, all_cols, index=idx, key=f"rm_{fld}")
            m_type   = _sel("Element Type",  'type',   mc[0])
            m_id     = _sel("ID / Mark",     'id',     mc[1])
            m_length = _sel("Length",        'length', mc[2])
            m_width  = _sel("Width",         'width',  mc[3])
            mc2 = st.columns(4)
            m_height = _sel("Height / Depth",'height', mc2[0])
            m_floor  = _sel("Level / Floor", 'floor',  mc2[1])
            m_zone   = _sel("Zone / Block",  'zone',   mc2[2])
            m_cost   = _sel("Cost (opt.)",   'cost',   mc2[3])

            cc = st.columns(3)
            unit        = cc[0].selectbox("Dimension units", ['mm','m','ft'])
            def_cost_rv = cc[1].number_input("Default cost/set", value=1500, step=100)
            def_date_rv = cc[2].text_input("Default cast date", value=datetime.today().strftime('%Y-%m-%d'))

            user_map = {k: (v if v != '(none)' else None) for k, v in [
                ('type',m_type),('id',m_id),('length',m_length),('width',m_width),
                ('height',m_height),('floor',m_floor),('zone',m_zone),('cost',m_cost)]}

            if st.button("▶ Convert Revit → SmartForm", type="primary", key="revit_go"):
                converted = convert_revit_to_smartform(
                    revit_raw, user_map, unit=unit,
                    default_cost=def_cost_rv, default_date=def_date_rv)
                if converted.empty:
                    st.error("0 rows produced — check column mapping.")
                else:
                    st.session_state['imported_df'] = converted
                    st.success(f"✅ {len(converted)} elements converted. Dashboard now uses this data.")
                    st.dataframe(converted.head(10).drop(columns=['_source'], errors='ignore'),
                                 use_container_width=True, hide_index=True)
                    st.download_button("⬇ Save as structural_elements.csv",
                                       data=converted.drop(columns=['_source'], errors='ignore').to_csv(index=False),
                                       file_name="structural_elements.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Parse error: {e}")

# ─ Tab 3: AutoCAD DXF ───────────────────────────────────────────────────────
with tab_dxf:
    if not EZDXF_OK:
        st.error("`ezdxf` not installed. Run `pip install ezdxf` then restart the app.")
    else:
        st.markdown(
            "💡 **AutoCAD export path:** File → Save As → AutoCAD DXF (.dxf).  \n"
            "Place structural elements on named layers, e.g. **S-COLS**, **S-SLAB**, **S-BEAM**."
        )
        dxf_file = st.file_uploader("Upload AutoCAD DXF file", type=['dxf'], key="dxf_upload")
        if dxf_file:
            import io as _io3
            dxf_bytes = dxf_file.read()
            layers = get_layers(_io3.BytesIO(dxf_bytes))

            if not layers or (layers and layers[0].startswith('ERROR')):
                st.error(f"Could not read DXF layers: {layers}")
            else:
                st.success(f"Found **{len(layers)} layers** in the DXF file.")
                st.markdown("**🔄 Assign each layer to an element type** (auto-guessed from layer name)")
                options   = ['(ignore)', 'Column', 'Slab', 'Beam']
                layer_map = {}
                n = min(len(layers), 4)
                cols4 = st.columns(n)
                for i, layer in enumerate(layers):
                    lw = layer.lower()
                    if any(k in lw for k in ['col','pillar','post']):    didx = 1
                    elif any(k in lw for k in ['slab','floor','deck']):  didx = 2
                    elif any(k in lw for k in ['beam','frm','girder']):  didx = 3
                    else:                                                 didx = 0
                    chosen = cols4[i % n].selectbox(layer, options, index=didx, key=f"dl_{i}")
                    if chosen != '(ignore)':
                        layer_map[layer] = chosen

                st.markdown("**📐 Default values per element type**")
                dc  = st.columns(4)
                dc2 = st.columns(4)
                dh_col   = dc[0].number_input("Column H (m)",    value=3.0,  step=0.1,  key='dhc')
                dh_slab  = dc[1].number_input("Slab thick (m)",  value=0.2,  step=0.05, key='dhs')
                dh_beam  = dc[2].number_input("Beam depth (m)",  value=0.5,  step=0.05, key='dhb')
                dw_beam  = dc[3].number_input("Beam width (m)",  value=0.3,  step=0.05, key='dwb')
                cost_col  = dc2[0].number_input("Col cost/set",   value=1200, step=100,  key='dcc')
                cost_slab = dc2[1].number_input("Slab cost/set",  value=3500, step=100,  key='dcs')
                cost_beam = dc2[2].number_input("Beam cost/set",  value=1800, step=100,  key='dcb')
                dxf_zone  = dc2[3].text_input("Zone", value='Zone-A', key='dxfz')
                dxf_date  = dc[0].text_input("Cast date (YYYY-MM-DD)",
                                              value=datetime.today().strftime('%Y-%m-%d'), key='dxfd')

                if st.button("▶ Parse DXF → SmartForm", type="primary",
                             disabled=not bool(layer_map), key="dxf_go"):
                    with st.spinner("Parsing DXF geometry…"):
                        parsed = parse_dxf(
                            _io3.BytesIO(dxf_bytes), layer_map=layer_map,
                            default_height={'Column':dh_col,'Slab':dh_slab,'Beam':dh_beam},
                            default_width={'Beam': dw_beam},
                            default_cost={'Column':cost_col,'Slab':cost_slab,'Beam':cost_beam},
                            default_date=dxf_date, default_zone=dxf_zone,
                        )
                    if parsed.empty:
                        st.warning("No elements extracted — verify layer assignments and that those layers have geometry.")
                    else:
                        st.session_state['imported_df'] = parsed
                        st.success(f"✅ Extracted **{len(parsed)} elements**. "
                                   f"Breakdown: {parsed['Type'].value_counts().to_dict()}")
                        st.dataframe(parsed.head(15).drop(columns=['_source'], errors='ignore'),
                                     use_container_width=True, hide_index=True)
                        st.download_button("⬇ Save as structural_elements.csv",
                                           data=parsed.drop(columns=['_source'], errors='ignore').to_csv(index=False),
                                           file_name="structural_elements.csv", mime="text/csv")

st.divider()


# ── Data resolution: import > sidebar upload > disk demo data ─────────────────
df_elements   = None
_source_label = ""

if st.session_state.get('imported_df') is not None:
    df_elements   = st.session_state['imported_df'].copy()
    _src          = df_elements['_source'].iloc[0].split(':')[0] if '_source' in df_elements.columns else 'Import'
    _source_label = f"Source: {_src} import — {len(df_elements)} elements"
    df_elements   = df_elements.drop(columns=['_source'], errors='ignore')
elif uploaded_elements is not None:
    df_elements   = load_data(uploaded_elements, _mtime=None)
    _source_label = f"Source: uploaded CSV — {len(df_elements)} elements"
else:
    df_elements   = load_data(None, _mtime=_file_mtime("structural_elements.csv"))
    if df_elements is not None:
        _source_label = f"Source: demo data — {len(df_elements)} elements"

if df_elements is None:
    st.info("Upload **structural_elements.csv** via the sidebar, import from Revit/DXF above, "
            "or run `python generate_mock_data.py` to generate demo data.")
    st.stop()

if _source_label:
    st.caption(_source_label)



# ── Active feature badges ─────────────────────────────────────────────────────
badges = []
badges.append(f"⏱ Cycle: {reuse_cycle}d ({curing_days}+{stripping_days}+{handling_days})")
if enable_life_limit:
    badges.append(f"♻️ Life limit: {max_reuse_count} uses")
if enable_zones and 'Zone' in df_elements.columns:
    badges.append("🗺 Zone splitting ON")
badges.append(f"📦 Lead time: {lead_time_days}d")

st.markdown(
    " &nbsp;·&nbsp; ".join(
        f"<span style='background:#EFF6FF;color:#1E40AF;border-radius:4px;"
        f"padding:3px 9px;font-size:12px;font-weight:600;'>{b}</span>"
        for b in badges
    ),
    unsafe_allow_html=True
)
st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)


# ── Processing pipeline ───────────────────────────────────────────────────────
df_elements = generate_boq(df_elements)
df_clustered, df_summary = detect_repetitions(df_elements)

zone_col = 'Zone' if (enable_zones and 'Zone' in df_clustered.columns) else None

with st.spinner("Running PuLP optimisation…"):
    df_opt = optimize_formwork_sets(
        df_clustered,
        reuse_cycle_days=reuse_cycle,
        max_reuse_count=max_reuse_count,
        zone_col=zone_col
    )

# ── KPI calculations ──────────────────────────────────────────────────────────
opt_cost        = df_opt['Optimized_Procurement_Cost'].sum()
naive_cost      = df_opt['Naive_Procurement_Cost'].sum()
proc_savings    = naive_cost - opt_cost
carry_savings   = proc_savings * carrying_cost_rate
total_savings   = proc_savings + carry_savings
proc_red_pct    = (proc_savings / naive_cost * 100) if naive_cost else 0
total_opt_sets  = df_opt['Required_Sets'].sum()
total_naive_sets = df_opt['Naive_Sets'].sum()
excess_red_pct  = ((total_naive_sets - total_opt_sets) / total_naive_sets * 100) if total_naive_sets else 0
write_off_total = df_opt['Write_Off_Cost'].sum() if 'Write_Off_Cost' in df_opt.columns else 0
true_total_cost = df_opt['True_Total_Cost'].sum() if 'True_Total_Cost' in df_opt.columns else opt_cost


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='font-size:10px;font-weight:700;letter-spacing:1.5px;"
    "text-transform:uppercase;color:#9CA3AF;margin-bottom:12px;'>Executive Summary</div>",
    unsafe_allow_html=True
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi_card(
        "Optimised Procurement Cost", f"₹{opt_cost:,.0f}",
        f"↓ {proc_red_pct:.1f}% vs no-reuse baseline", "#1565C0"
    ), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card(
        "Total Projected Savings", f"₹{total_savings:,.0f}",
        "Capex + carrying costs combined", "#00695C"
    ), unsafe_allow_html=True)
with c3:
    label = f"₹{write_off_total:,.0f}" if enable_life_limit else f"{total_opt_sets} sets"
    sub   = f"Write-off cost ({df_opt['Sets_Written_Off'].sum() if enable_life_limit else ''}{' sets scrapped)' if enable_life_limit else f'↓ {excess_red_pct:.1f}% vs naive'}"
    st.markdown(kpi_card(
        "Write-off Cost" if enable_life_limit else "Optimised Inventory",
        label, sub, "#E65100"
    ), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card(
        "True Total Cost" if enable_life_limit else "Carrying Cost Saved",
        f"₹{true_total_cost:,.0f}" if enable_life_limit else f"₹{carry_savings:,.0f}",
        f"Procurement + write-off" if enable_life_limit else f"At {carrying_cost_rate*100:.0f}% carrying rate",
        "#6A1B9A"
    ), unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 01 — Repetition Detection
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(1, "Repetition Detection & BoQ Clusters",
                           "Elements sharing identical type + dimensions share the same formwork set"),
            unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    st.caption("Top 10 clusters by repetition frequency")
    st.dataframe(df_summary.head(10), use_container_width=True, hide_index=True)
with col_b:
    fig_rep, ax = plt.subplots(figsize=(8, 4.2))
    top10   = df_summary.head(10)
    colours = ["#1565C0"] * len(top10)
    if colours: colours[0] = "#0D47A1"
    bars = ax.barh(top10['Cluster_ID'][::-1], top10['Frequency'][::-1],
                   color=colours[::-1], height=0.6, edgecolor='none')
    for bar, val in zip(bars, top10['Frequency'][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=9, color='#444')
    ax.set_xlabel("Repetition Count", fontsize=10)
    ax.set_title("Element Repetition by Cluster", fontsize=12,
                 fontweight='bold', color='#1A1A2E', pad=10)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, top10['Frequency'].max() * 1.12)
    plt.tight_layout()
    st.pyplot(fig_rep)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 02 — Optimisation Results
# (Feature 1: Write-off columns · Feature 2: Zone column)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(2, "Procurement Optimisation Results",
                           f"PuLP LP · {reuse_cycle}-day effective cycle"
                           + (f" · Max {max_reuse_count} uses/set" if enable_life_limit else "")
                           + (" · Zone-split inventory" if zone_col else "")),
            unsafe_allow_html=True)

# Choose columns to show based on active features
base_cols = ['Cluster_ID', 'Zone', 'Required_Sets', 'Naive_Sets',
             'Optimized_Procurement_Cost', 'Cost_Savings', 'Cost_Reduction_%']
if enable_life_limit:
    base_cols += ['Sets_Written_Off', 'Write_Off_Cost', 'True_Total_Cost']
if not zone_col:
    base_cols = [c for c in base_cols if c != 'Zone']

display_df = df_opt[[c for c in base_cols if c in df_opt.columns]].copy()
rename_map = {
    'Cluster_ID':                 'Cluster',
    'Zone':                       'Zone',
    'Required_Sets':              'Opt. Sets',
    'Naive_Sets':                 'Naive Sets',
    'Optimized_Procurement_Cost': 'Opt. Cost (₹)',
    'Cost_Savings':               'Savings (₹)',
    'Cost_Reduction_%':           'Reduction %',
    'Sets_Written_Off':           'Write-offs',
    'Write_Off_Cost':             'Write-off Cost (₹)',
    'True_Total_Cost':            'True Total (₹)',
}
display_df.rename(columns=rename_map, inplace=True)

fmt = {
    'Opt. Cost (₹)':     '₹{:,.0f}',
    'Savings (₹)':       '₹{:,.0f}',
    'Reduction %':       '{:.1f}%',
    'Write-off Cost (₹)':'₹{:,.0f}',
    'True Total (₹)':   '₹{:,.0f}',
}
st.dataframe(
    display_df.style
              .format({k: v for k, v in fmt.items() if k in display_df.columns})
              .background_gradient(subset=['Savings (₹)'] if 'Savings (₹)' in display_df.columns else [], cmap='YlGn')
              .set_properties(**{'font-size': '13px'}),
    use_container_width=True,
    hide_index=True,
    height=300
)

# Summary metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Optimised Sets",  f"{total_opt_sets:,}",
          f"-{total_naive_sets - total_opt_sets:,} vs naive")
m2.metric("Procurement Reduction", f"₹{proc_savings:,.0f}",
          f"-{proc_red_pct:.1f}%")
if enable_life_limit:
    m3.metric("Sets Written Off", f"{df_opt['Sets_Written_Off'].sum():,}")
    m4.metric("Total Write-off Cost", f"₹{write_off_total:,.0f}")
else:
    m3.metric("Avg Savings / Cluster",
              f"₹{(proc_savings / len(df_opt) if len(df_opt) else 0):,.0f}")
    m4.metric("Carrying Cost Saved", f"₹{carry_savings:,.0f}",
              f"@ {carrying_cost_rate*100:.0f}%")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 03 — Inventory Timeline
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(3, "Inventory Simulation Timeline",
                           "Daily in-use sets vs total optimised inventory over the project lifespan"),
            unsafe_allow_html=True)

df_timeline = simulate_timeline(df_clustered, df_opt, reuse_cycle_days=reuse_cycle)
reuse_eff   = (
    df_timeline['Required Sets (Active)'].sum() /
    (total_opt_sets * len(df_timeline)) * 100
) if total_opt_sets else 0

st.markdown(
    f"<p style='font-size:13px;color:#6B7280;margin-bottom:14px;'>"
    f"Average daily reuse efficiency: "
    f"<strong style='color:#1A1A2E;'>{reuse_eff:.1f}%</strong></p>",
    unsafe_allow_html=True
)

fig_tl, ax2 = plt.subplots(figsize=(12, 4.2))
ax2.fill_between(df_timeline['Date'], df_timeline['Required Sets (Active)'],
                 alpha=0.12, color='#1565C0')
ax2.plot(df_timeline['Date'], df_timeline['Available Sets (Inventory)'],
         label='Inventory Available', color='#B0BEC5', linewidth=1.8,
         linestyle='--', dashes=(5, 3))
ax2.plot(df_timeline['Date'], df_timeline['Required Sets (Active)'],
         label='In-Use (Active)', color='#1565C0', linewidth=2.2)
ax2.set_xlabel("Project Date", fontsize=10)
ax2.set_ylabel("Formwork Sets", fontsize=10)
ax2.set_title("Formwork Usage vs Available Inventory", fontsize=12,
              fontweight='bold', color='#1A1A2E', pad=10)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=10, framealpha=0.9)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.tight_layout()
st.pyplot(fig_tl)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 04 — Procurement Schedule  (Feature 4)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(4, "Procurement Schedule",
                           f"Order-by dates based on {lead_time_days}-day supplier lead time"),
            unsafe_allow_html=True)

df_proc = generate_procurement_schedule(df_opt, df_clustered, lead_time_days=lead_time_days)

urgent_count = (df_proc['Status'] == '🔴 URGENT').sum()
soon_count   = (df_proc['Status'] == '🟡 ORDER SOON').sum()

alert_cols = st.columns(3)
alert_cols[0].metric("🔴 Urgent Orders",    urgent_count, help="Order date already passed")
alert_cols[1].metric("🟡 Order Soon",       soon_count,   help="Order date within lead time window")
alert_cols[2].metric("🟢 Planned",          len(df_proc) - urgent_count - soon_count)

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# Procurement table
proc_display = df_proc.copy()
if not zone_col:
    proc_display = proc_display.drop(columns=['Zone'], errors='ignore')
if 'Zone' in proc_display.columns and proc_display['Zone'].nunique() == 1:
    proc_display = proc_display.drop(columns=['Zone'])

proc_display_show = proc_display[[c for c in [
    'Cluster_ID', 'Zone', 'Sets_To_Order', 'First_Pour_Date',
    'Order_By_Date', 'Days_Until_Order', 'Estimated_Cost', 'Status'
] if c in proc_display.columns]].rename(columns={
    'Cluster_ID':      'Cluster',
    'Sets_To_Order':   'Qty',
    'First_Pour_Date': 'First Pour',
    'Order_By_Date':   'Order By',
    'Days_Until_Order':'Days Left',
    'Estimated_Cost':  'Est. Cost (₹)',
})

def _color_status(val):
    if 'URGENT' in str(val):   return 'color: #B91C1C; font-weight:700'
    if 'ORDER SOON' in str(val): return 'color: #B45309; font-weight:700'
    return 'color: #166534; font-weight:600'

st.dataframe(
    proc_display_show.style
        .applymap(_color_status, subset=['Status'])
        .format({'Est. Cost (₹)': '₹{:,.0f}'})
        .set_properties(**{'font-size': '13px'}),
    use_container_width=True,
    hide_index=True,
    height=320
)

# Procurement Gantt-style chart
st.markdown("**Order → Delivery → First Pour timeline**")
fig_gantt, ax_g = plt.subplots(figsize=(12, min(len(df_proc) * 0.45 + 1.5, 10)))

colours_map = {'🔴 URGENT': '#DC2626', '🟡 ORDER SOON': '#D97706', '🟢 PLANNED': '#16A34A'}

for idx, row in df_proc.head(20).iterrows():
    order_date    = pd.Timestamp(row['Order_By_Date'])
    pour_date     = pd.Timestamp(row['First_Pour_Date'])
    deliver_date  = order_date + pd.Timedelta(days=lead_time_days)
    colour        = colours_map.get(row['Status'], '#6B7280')
    label         = f"{row['Cluster_ID']}" + (f" / {row['Zone']}" if 'Zone' in row and zone_col else "")

    # Lead-time bar (order → delivery)
    ax_g.barh(label, (deliver_date - order_date).days, left=order_date,
              color=colour, alpha=0.5, height=0.5)
    # Buffer bar (delivery → first pour)
    ax_g.barh(label, (pour_date - deliver_date).days, left=deliver_date,
              color=colour, alpha=0.2, height=0.5)
    # Pour marker
    ax_g.scatter(pour_date, label, marker='D', color=colour, s=40, zorder=5)

ax_g.set_xlabel("Date", fontsize=10)
ax_g.set_title("Procurement Gantt (Order → Delivery → First Pour) — Top 20 Clusters",
               fontsize=11, fontweight='bold', color='#1A1A2E', pad=8)
ax_g.spines[['top', 'right']].set_visible(False)
ax_g.xaxis_date()
fig_gantt.autofmt_xdate(rotation=30)
plt.tight_layout()
st.pyplot(fig_gantt)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Final Summary Banner
# ══════════════════════════════════════════════════════════════════════════════
feature_tags = " &nbsp;·&nbsp; ".join([
    "Feature 3: Advanced Reuse Cycle",
    "Feature 4: Procurement Scheduler",
    *(["Feature 1: Reuse Life Limit"] if enable_life_limit else []),
    *(["Feature 2: Zone Splitting"] if zone_col else []),
])
st.markdown(f"""
<div style="background:#1A1A2E;border-radius:12px;padding:26px 30px;margin-top:6px;">
  <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:#7986CB;margin-bottom:8px;">PROJECT FINAL SUMMARY</div>
  <div style="font-size:20px;font-weight:800;color:#FFFFFF;margin-bottom:4px;">
    ₹{total_savings:,.0f} estimated total savings
  </div>
  <div style="font-size:12px;color:#9FA8DA;margin-top:4px;">
    {excess_red_pct:.1f}% excess inventory reduction &nbsp;·&nbsp;
    {reuse_eff:.1f}% reuse efficiency &nbsp;·&nbsp;
    {reuse_cycle}-day effective cycle
    {f" &nbsp;·&nbsp; {df_opt['Sets_Written_Off'].sum()} sets written off" if enable_life_limit else ""}
  </div>
  <div style="font-size:10px;color:#5C6BC0;margin-top:12px;">{feature_tags}</div>
</div>
""", unsafe_allow_html=True)
