"""
revit_importer.py
Parses Autodesk Revit schedule exports (.csv or .xlsx) and maps columns
to SmartForm AI's structural_elements format.
"""
import pandas as pd
import io
from datetime import datetime


# ── Revit AIA layer/family name → SmartForm type ─────────────────────────────
_TYPE_KEYWORDS = {
    'column':  'Column',
    'col':     'Column',
    'pillar':  'Column',
    'post':    'Column',
    'slab':    'Slab',
    'floor':   'Slab',
    'deck':    'Slab',
    'flat':    'Slab',
    'beam':    'Beam',
    'framing': 'Beam',
    'girder':  'Beam',
    'joist':   'Beam',
    'lintel':  'Beam',
}

def _infer_type(value: str) -> str:
    v = str(value).lower()
    for kw, t in _TYPE_KEYWORDS.items():
        if kw in v:
            return t
    return 'Column'   # safe default

def _to_metres(value, unit='mm'):
    """Convert a dimension value to metres."""
    try:
        v = float(str(value).replace(',', '').strip())
    except (ValueError, TypeError):
        return None
    if unit == 'mm':
        return round(v / 1000, 4)
    if unit == 'ft':
        return round(v * 0.3048, 4)
    return round(v, 4)          # assume already metres


def load_revit_file(file_obj, filename: str) -> pd.DataFrame:
    """
    Load a raw Revit schedule export.
    Handles:
    - CSV with 1-2 header rows (schedule title row + blank row before column headers)
    - XLSX from Revit's 'Export Schedule' option
    Returns a raw DataFrame with original Revit column names.
    """
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # Try to find the real header row (first row with more than 2 non-null values)
        raw = pd.read_excel(file_obj, header=None)
        header_row = 0
        for i, row in raw.iterrows():
            if row.notna().sum() >= 3:
                header_row = i
                break
        df = pd.read_excel(io.BytesIO(file_obj.getvalue()) if hasattr(file_obj, 'getvalue') else file_obj,
                           header=header_row)
    else:
        # CSV — skip schedule-name rows until real headers found
        raw_text = file_obj.read().decode('utf-8', errors='replace') \
            if hasattr(file_obj, 'read') else open(file_obj, encoding='utf-8').read()
        lines = raw_text.splitlines()
        header_idx = 0
        for i, line in enumerate(lines):
            cols = [c.strip() for c in line.split(',') if c.strip()]
            if len(cols) >= 3:
                header_idx = i
                break
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])))

    df.columns = [str(c).strip() for c in df.columns]
    return df


def auto_map_columns(df: pd.DataFrame) -> dict:
    """
    Attempt to auto-detect which Revit columns map to SmartForm fields.
    Returns a dict: {'type': col, 'length': col, 'width': col, ...}
    """
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}

    def find(keywords):
        for kw in keywords:
            for cl, orig in cols_lower.items():
                if kw in cl:
                    return orig
        return None

    mapping['type']   = find(['family', 'type', 'category', 'element'])
    mapping['length'] = find(['length', 'len', 'span'])
    mapping['width']  = find(['width', 'wid', 'b', 'breadth'])
    mapping['height'] = find(['height', 'depth', 'thickness', 'thk', 'h', 'depth'])
    mapping['floor']  = find(['level', 'floor', 'storey', 'story'])
    mapping['date']   = find(['cast', 'pour', 'date', 'constr'])
    mapping['cost']   = find(['cost', 'rate', 'price'])
    mapping['zone']   = find(['zone', 'block', 'wing', 'sector'])
    mapping['id']     = find(['mark', 'id', 'tag', 'no', 'number', 'ref'])

    return mapping


def convert_revit_to_smartform(
    df: pd.DataFrame,
    mapping: dict,
    unit: str = 'mm',
    default_cost: float = 1500.0,
    default_date: str = None,
    default_zone: str = 'Zone-A',
) -> pd.DataFrame:
    """
    Apply the column mapping and produce a SmartForm-ready DataFrame.

    Parameters
    ----------
    df           : Raw Revit DataFrame
    mapping      : Dict from auto_map_columns() (user can override in UI)
    unit         : Dimension unit in Revit file: 'mm', 'm', or 'ft'
    default_cost : Fallback formwork cost if no cost column mapped
    default_date : Fallback casting date (YYYY-MM-DD); today if None
    default_zone : Fallback zone label
    """
    if default_date is None:
        default_date = datetime.today().strftime('%Y-%m-%d')

    rows = []
    for idx, row in df.iterrows():
        def get(field, fallback=None):
            col = mapping.get(field)
            return row[col] if (col and col in row.index and pd.notna(row[col])) else fallback

        raw_type = get('type', 'Column')
        elem_type = _infer_type(raw_type)

        length = _to_metres(get('length', 0.5 if elem_type == 'Column' else 5.0), unit)
        width  = _to_metres(get('width',  0.5 if elem_type == 'Column' else 0.3), unit)
        height = _to_metres(get('height', 3.0 if elem_type != 'Slab'   else 0.2), unit)

        if None in (length, width, height):
            continue

        # Floor: try to extract numeric part
        floor_raw = str(get('floor', '1'))
        import re
        floor_nums = re.findall(r'\d+', floor_raw)
        floor = int(floor_nums[0]) if floor_nums else 1

        zone  = str(get('zone', default_zone)).strip() or default_zone
        date  = str(get('date', default_date)).strip()[:10]
        cost  = float(get('cost', default_cost) or default_cost)
        elem_id = str(get('id', f"RVT-{idx+1:03d}"))

        rows.append({
            'Element_ID':             elem_id,
            'Type':                   elem_type,
            'Length':                 length,
            'Width':                  width,
            'Height':                 height,
            'Floor':                  floor,
            'Zone':                   zone,
            'Casting_Date':           date,
            'Formwork_Cost_per_Set':  cost,
            'Replacement_Cost_per_Set': round(cost * 0.85, 0),
            'Max_Reuse_Count':        10,
            '_source':                'Revit',
        })

    return pd.DataFrame(rows)
