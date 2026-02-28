"""
kitting_engine.py
Formwork Kitting — assigns physical elements to specific formwork kits
and generates a detailed kit utilization plan.

Kitting:
  A "kit" is one physical formwork set. Each kit is assigned to a sequence
  of elements that share the same dimensions (same Cluster_ID) and are
  scheduled with enough gap between them to allow reuse (reuse_cycle_days).

Output:
  - kit_plan DataFrame: Element_ID → Kit_ID, Kit_No, cast_date, reuse_count
  - kit_summary DataFrame: per Kit_ID stats
  - Standardization Score: 0–100 reflecting design repetition quality
"""
import pandas as pd
import numpy as np
from datetime import timedelta


def assign_kits(df_elements: pd.DataFrame, reuse_cycle_days: int = 7) -> pd.DataFrame:
    """
    Assign each element to a specific kit (physical formwork set).

    Strategy:
      Within each cluster (same dimensions), sort elements by Casting_Date.
      Greedily assign each element to the earliest kit that has been free for
      ≥ reuse_cycle_days since its last use. If none available, create a new kit.

    Parameters
    ----------
    df_elements       : Clustered element DataFrame (must have Cluster_ID, Casting_Date).
    reuse_cycle_days  : Days a kit is locked after a pour.

    Returns
    -------
    df_elements with added columns:
        Kit_ID        – unique physical kit identifier (e.g. "Column_0.5x0.5_Kit-01")
        Kit_Number    – integer kit number within its cluster (1-indexed)
        Reuse_Count   – how many times this kit has been used up to this pour
        Days_Since_Prev – days since last use of this kit (0 for first use)
    """
    df = df_elements.copy()
    df['Casting_Date'] = pd.to_datetime(df['Casting_Date'])
    df = df.sort_values(['Cluster_ID', 'Casting_Date']).reset_index(drop=True)

    kit_id_col      = []
    kit_no_col      = []
    reuse_count_col = []
    days_since_col  = []

    for cluster, grp in df.groupby('Cluster_ID', sort=False):
        # Kit tracker: kit_no → last_used_date
        kit_last_used: dict = {}   # kit_no (int) → last Casting_Date (pd.Timestamp)
        kit_reuse_count: dict = {} # kit_no → int

        assignments = {}  # row index → (kit_no, reuse_count, days_since)

        for idx, row in grp.iterrows():
            cast = row['Casting_Date']
            assigned = None

            # Try to find an available kit (cured for ≥ reuse_cycle_days)
            for kit_no, last_used in sorted(kit_last_used.items()):
                gap = (cast - last_used).days
                if gap >= reuse_cycle_days:
                    assigned   = kit_no
                    days_since = gap
                    break

            if assigned is None:
                # All existing kits still curing — create a new one
                assigned   = len(kit_last_used) + 1
                days_since = 0

            kit_last_used[assigned] = cast
            kit_reuse_count[assigned] = kit_reuse_count.get(assigned, 0) + 1
            assignments[idx] = (assigned, kit_reuse_count[assigned], days_since)

        for idx, (kit_no, reuse_cnt, days) in assignments.items():
            # Build human-readable kit ID
            row = df.loc[idx]
            size = f"{row['Length']}x{row['Width']}" if 'Length' in df.columns else ''
            kid  = f"{cluster}_Kit-{kit_no:02d}"
            kit_id_col.append(kid)
            kit_no_col.append(kit_no)
            reuse_count_col.append(reuse_cnt)
            days_since_col.append(days)

    df['Kit_ID']         = kit_id_col
    df['Kit_Number']     = kit_no_col
    df['Reuse_Count']    = reuse_count_col
    df['Days_Since_Prev']= days_since_col

    return df.sort_values(['Casting_Date', 'Cluster_ID']).reset_index(drop=True)


def kit_summary(df_kitted: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per Kit_ID with utilisation stats.
    """
    grp = df_kitted.groupby('Kit_ID').agg(
        Cluster_ID    = ('Cluster_ID', 'first'),
        Type          = ('Type', 'first'),
        Total_Uses    = ('Kit_ID', 'count'),
        First_Use     = ('Casting_Date', 'min'),
        Last_Use      = ('Casting_Date', 'max'),
        Elements      = ('Element_ID', lambda x: ', '.join(x.astype(str))),
    ).reset_index()

    grp['Active_Days'] = (
        pd.to_datetime(grp['Last_Use']) - pd.to_datetime(grp['First_Use'])
    ).dt.days + 1

    grp['Reuse_Efficiency_%'] = ((grp['Total_Uses'] - 1) / grp['Total_Uses'] * 100).round(1)

    return grp.sort_values(['Cluster_ID', 'Kit_ID']).reset_index(drop=True)


def standardization_score(df_elements: pd.DataFrame) -> dict:
    """
    Compute a Standardization Score (0–100) reflecting design repetition quality.

    Formula:
      Score = (1 - unique_variants / total_elements) × repetition_bonus × 100

    Components:
      - Unique type-dimension variants vs total elements
      - Average elements per cluster (higher = better repetition)
      - Cluster concentration: top cluster's share of total
    """
    if df_elements.empty:
        return {'score': 0, 'grade': 'N/A', 'details': {}}

    total     = len(df_elements)
    n_clusters = df_elements['Cluster_ID'].nunique() if 'Cluster_ID' in df_elements.columns else total

    # Component 1: repetition ratio (0-1)
    rep_ratio  = 1 - (n_clusters / total)   # 1 = all elements identical, 0 = all unique

    # Component 2: average cluster size (normalize to 1 for clusters of 5+)
    avg_size   = total / n_clusters
    size_bonus = min(avg_size / 10, 1.0)    # saturates at avg cluster size 10

    # Component 3: type diversity penalty (having all 3 types = good structural variety)
    n_types    = df_elements['Type'].nunique() if 'Type' in df_elements.columns else 1
    type_bonus = min(n_types / 3, 1.0)

    raw_score  = (rep_ratio * 0.6 + size_bonus * 0.3 + type_bonus * 0.1) * 100
    score      = round(min(max(raw_score, 0), 100), 1)

    if score >= 80:   grade, colour = 'A — Excellent',   '#16A34A'
    elif score >= 65: grade, colour = 'B — Good',        '#65A30D'
    elif score >= 50: grade, colour = 'C — Average',     '#D97706'
    elif score >= 35: grade, colour = 'D — Below Avg',   '#EA580C'
    else:             grade, colour = 'F — Poor',        '#DC2626'

    details = {
        'total_elements':    total,
        'unique_clusters':   n_clusters,
        'avg_cluster_size':  round(avg_size, 1),
        'repetition_ratio':  round(rep_ratio * 100, 1),
        'type_variety':      n_types,
    }
    return {'score': score, 'grade': grade, 'colour': colour, 'details': details}


def export_to_excel(
    df_elements: pd.DataFrame,
    df_boq: pd.DataFrame,
    df_kitted: pd.DataFrame,
    df_kit_summary: pd.DataFrame,
    df_opt: pd.DataFrame,
    df_proc: pd.DataFrame,
    std_score: dict,
    project_name: str = "SmartForm AI Report"
) -> bytes:
    """
    Write all results to an Excel workbook and return as bytes for download.
    """
    import io
    try:
        import xlsxwriter
        engine = 'xlsxwriter'
    except ImportError:
        engine = 'openpyxl'

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=engine) as writer:
        # Sheet 1: Summary
        summary_data = {
            'Metric': [
                'Total Elements', 'Unique Clusters', 'Optimised Sets Required',
                'Naive Sets (no reuse)', 'Procurement Cost Savings (₹)',
                'Standardization Score', 'Standardization Grade'
            ],
            'Value': [
                len(df_elements),
                df_elements['Cluster_ID'].nunique() if 'Cluster_ID' in df_elements.columns else '-',
                df_opt['Required_Sets'].sum() if not df_opt.empty else '-',
                df_opt['Naive_Sets'].sum()    if not df_opt.empty else '-',
                f"₹{df_opt['Cost_Savings'].sum():,.0f}" if not df_opt.empty else '-',
                std_score.get('score', '-'),
                std_score.get('grade', '-'),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        df_elements.to_excel(writer,     sheet_name='Structural Elements', index=False)
        if not df_boq.empty:
            df_boq.to_excel(writer,      sheet_name='BoQ',                 index=False)
        df_kitted.to_excel(writer,       sheet_name='Kitting Plan',        index=False)
        df_kit_summary.to_excel(writer,  sheet_name='Kit Summary',         index=False)
        df_opt.to_excel(writer,          sheet_name='Optimisation',        index=False)
        df_proc.to_excel(writer,         sheet_name='Procurement Schedule', index=False)

    return buf.getvalue()


def export_kitting_plan_excel(
    df_kitted: pd.DataFrame,
    df_kit_summ: pd.DataFrame,
    reuse_cycle_days: int = 7,
    project_name: str = "Project",
) -> bytes:
    """
    Generate a richly formatted, site-ready Kitting Plan Excel workbook.

    Sheets:
      1. README         — how to read this file
      2. Chronological Flow  — every pour in date order: Frame → Element ID
      3. Frame Register — one row per physical kit, all element IDs listed in sequence
      4. By Element Type — separate tables for Columns, Slabs, Beams
    """
    import io
    buf = io.BytesIO()

    try:
        import xlsxwriter as _xl
        engine = 'xlsxwriter'
    except ImportError:
        engine = 'openpyxl'

    # ── Prepare Chronological Flow sheet ──────────────────────────────────────
    flow_cols = [c for c in [
        'Casting_Date', 'Kit_ID', 'Element_ID', 'Type', 'Floor', 'Zone',
        'Length', 'Width', 'Height', 'Cluster_ID', 'Kit_Number',
        'Reuse_Count', 'Days_Since_Prev'
    ] if c in df_kitted.columns]

    flow = df_kitted[flow_cols].copy()
    flow = flow.sort_values('Casting_Date').reset_index(drop=True)
    flow.index = flow.index + 1   # 1-based row numbers
    flow.insert(0, 'Pour #', flow.index)
    flow.rename(columns={
        'Kit_ID':        'Frame ID',
        'Element_ID':    'Element ID',
        'Reuse_Count':   'Reuse #',
        'Days_Since_Prev': 'Gap Since Last Use (days)',
        'Kit_Number':    'Frame No.',
    }, inplace=True)

    # ── Prepare Frame Register sheet ───────────────────────────────────────────
    reg_rows = []
    for kit_id, grp in df_kitted.groupby('Kit_ID'):
        grp_sorted = grp.sort_values('Casting_Date')
        for use_n, (_, row) in enumerate(grp_sorted.iterrows(), 1):
            reg_rows.append({
                'Frame ID':           kit_id,
                'Element Type':       row.get('Type', ''),
                'Dimensions (LxWxH)': (f"{row.get('Length','?')}×"
                                       f"{row.get('Width','?')}×"
                                       f"{row.get('Height','?')} m"),
                'Use #':              use_n,
                'Element ID':         row.get('Element_ID', ''),
                'Floor':              row.get('Floor', ''),
                'Zone':               row.get('Zone', ''),
                'Casting Date':       str(row.get('Casting_Date', ''))[:10],
                'Gap (days)':         int(row.get('Days_Since_Prev', 0)),
                'Status':             'REUSE' if use_n > 1 else 'FIRST USE',
            })
    df_reg = pd.DataFrame(reg_rows)

    # ── Prepare Kit Summary sheet ──────────────────────────────────────────────
    summ = df_kit_summ.rename(columns={
        'Kit_ID': 'Frame ID', 'Total_Uses': 'Total Uses',
        'First_Use': 'First Use', 'Last_Use': 'Last Use',
        'Active_Days': 'Active Days', 'Reuse_Efficiency_%': 'Efficiency %',
        'Elements': 'Element IDs Used',
    })

    # ── README content ─────────────────────────────────────────────────────────
    readme_rows = [
        ['SmartForm AI — Formwork Kitting Plan', ''],
        ['Project:', project_name],
        ['Reuse Cycle:', f'{reuse_cycle_days} days'],
        ['', ''],
        ['HOW TO USE THIS FILE', ''],
        ['', ''],
        ['Sheet: Chronological Flow',
         'Shows every concrete pour in date order. '
         '"Frame ID" is the physical formwork set to send to site that day. '
         '"Pour #" is the sequence number across the whole project.'],
        ['Sheet: Frame Register',
         'One row per use of each physical frame. '
         'Use this as a dispatch log — tick off each row as the frame leaves the yard.'],
        ['Sheet: Kit Summary',
         'One row per Frame ID — how many times it was reused, total active days, '
         'and efficiency %. High efficiency = good design standardisation.'],
        ['Sheet: By Element Type',
         'Columns, Slabs and Beams separated for clarity.'],
        ['', ''],
        ['GLOSSARY', ''],
        ['Frame ID',      'Unique name for one physical formwork set, e.g. CL-001_Kit-02'],
        ['Element ID',    'The structural element being cast (maps back to your drawing number)'],
        ['Reuse #',       'How many times this frame has been used up to and including this pour'],
        ['Gap (days)',     'Days since this frame was last used. Must be ≥ reuse cycle.'],
        ['FIRST USE',     'Frame is brand new for this pour'],
        ['REUSE',         'Frame was previously used on an earlier element'],
    ]
    df_readme = pd.DataFrame(readme_rows, columns=['Item', 'Description'])

    # ── Write to Excel ─────────────────────────────────────────────────────────
    with pd.ExcelWriter(buf, engine=engine) as writer:

        df_readme.to_excel(writer, sheet_name='README', index=False)
        flow.to_excel(writer, sheet_name='Chronological Flow', index=False)
        df_reg.to_excel(writer, sheet_name='Frame Register', index=False)
        summ.to_excel(writer, sheet_name='Kit Summary', index=False)

        # By Element Type
        for etype, colour in [('Column','#DBEAFE'), ('Slab','#DCFCE7'), ('Beam','#FEF9C3')]:
            sub = flow[flow['Type'] == etype] if 'Type' in flow.columns else pd.DataFrame()
            if not sub.empty:
                sub.to_excel(writer, sheet_name=f'{etype}s', index=False)

        # ── xlsxwriter formatting ──────────────────────────────────────────────
        if engine == 'xlsxwriter':
            wb = writer.book
            type_colours = {'Column': '#BFDBFE', 'Slab': '#BBF7D0', 'Beam': '#FDE68A'}
            hdr_fmt = wb.add_format({
                'bold': True, 'bg_color': '#1E3A5F', 'font_color': 'white',
                'border': 1, 'align': 'center', 'valign': 'vcenter',
                'text_wrap': True
            })
            reuse_fmt = wb.add_format({'bg_color': '#D1FAE5', 'font_color': '#065F46'})
            first_fmt = wb.add_format({'bg_color': '#EFF6FF', 'font_color': '#1E40AF'})
            urgent_fmt= wb.add_format({'bold': True, 'font_color': '#B91C1C'})

            # Format Chronological Flow
            ws_flow = writer.sheets['Chronological Flow']
            ws_flow.set_column('A:A', 8)   # Pour #
            ws_flow.set_column('B:B', 12)  # Casting Date
            ws_flow.set_column('C:C', 30)  # Frame ID
            ws_flow.set_column('D:D', 14)  # Element ID
            ws_flow.set_column('E:E', 10)  # Type
            for col_n, col_name in enumerate(flow.columns):
                ws_flow.write(0, col_n, col_name, hdr_fmt)
            ws_flow.freeze_panes(1, 0)
            ws_flow.autofilter(0, 0, len(flow), len(flow.columns) - 1)

            # Format Frame Register
            ws_reg = writer.sheets['Frame Register']
            for col_n, col_name in enumerate(df_reg.columns):
                ws_reg.write(0, col_n, col_name, hdr_fmt)
            ws_reg.set_column('A:A', 30)
            ws_reg.set_column('C:C', 20)
            ws_reg.set_column('I:I', 10)
            ws_reg.freeze_panes(1, 0)
            ws_reg.autofilter(0, 0, len(df_reg), len(df_reg.columns) - 1)
            # Colour-code REUSE vs FIRST USE
            status_col = df_reg.columns.get_loc('Status')
            for row_n, status in enumerate(df_reg['Status'], 1):
                fmt = reuse_fmt if status == 'REUSE' else first_fmt
                ws_reg.set_row(row_n, None, fmt)

    return buf.getvalue()
