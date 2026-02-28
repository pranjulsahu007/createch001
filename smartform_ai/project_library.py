"""
project_library.py
Manages a local library of historical project CSV files so the estimator
can learn from them and predict BoQ for new projects.

Storage layout:
  project_library/
    registry.json          — metadata for all saved projects
    <project_id>/
      structural_elements.csv
"""
import os, json, shutil, hashlib, re
import pandas as pd
from datetime import datetime

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "project_library")
REGISTRY    = os.path.join(LIBRARY_DIR, "registry.json")


def _ensure_dirs():
    os.makedirs(LIBRARY_DIR, exist_ok=True)


def _load_registry() -> list:
    _ensure_dirs()
    if os.path.exists(REGISTRY):
        with open(REGISTRY, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle old dict format → convert to list
            if isinstance(data, dict):
                return list(data.values())
            return data
    return []


def _save_registry(reg: list):
    _ensure_dirs()
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def _make_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]
    ts   = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{slug}_{ts}"


# ── Public API ─────────────────────────────────────────────────────────────────

def save_project(
    name:           str,
    building_type:  str,
    city:           str,
    floors:         int,
    floor_area_m2:  float,
    duration_days:  int,
    df_elements:    pd.DataFrame,
    notes:          str = "",
) -> str:
    """
    Save a project to the library. Returns the project_id.
    """
    reg       = _load_registry()
    proj_id   = _make_id(name)
    proj_dir  = os.path.join(LIBRARY_DIR, proj_id)
    os.makedirs(proj_dir, exist_ok=True)

    csv_path  = os.path.join(proj_dir, "structural_elements.csv")
    df_elements.to_csv(csv_path, index=False)

    # Derive summary stats for fast access during training
    def _counts(df):
        vc = df['Type'].value_counts().to_dict()
        return {
            'n_columns': int(vc.get('Column', 0)),
            'n_slabs':   int(vc.get('Slab',   0)),
            'n_beams':   int(vc.get('Beam',   0)),
            'n_total':   len(df),
        }

    def _avg_dims(df, etype):
        sub = df[df['Type'] == etype]
        if sub.empty:
            return {'length': 0, 'width': 0, 'height': 0}
        return {
            'length': round(sub['Length'].mean(), 3),
            'width':  round(sub['Width'].mean(),  3),
            'height': round(sub['Height'].mean(), 3),
        }

    def _total_cost(df):
        if 'Formwork_Cost_per_Set' in df.columns:
            return float(df['Formwork_Cost_per_Set'].sum())
        return 0.0

    def _n_clusters(df):
        """Count unique dimension groups from the actual CSV data."""
        dim_cols = [c for c in ['Type', 'Length', 'Width', 'Height'] if c in df.columns]
        if not dim_cols:
            return len(df)
        return int(df[dim_cols].drop_duplicates().shape[0])

    counts = _counts(df_elements)
    entry = {
        'id':            proj_id,
        'name':          name,
        'building_type': building_type,
        'city':          city,
        'floors':        floors,
        'floor_area_m2': floor_area_m2,
        'duration_days': duration_days,
        'notes':         notes,
        'saved_at':      datetime.now().isoformat(),
        'csv_path':      csv_path,
        **counts,
        'n_clusters':    _n_clusters(df_elements),
        'col_dims':      _avg_dims(df_elements, 'Column'),
        'slab_dims':     _avg_dims(df_elements, 'Slab'),
        'beam_dims':     _avg_dims(df_elements, 'Beam'),
        'total_cost':    _total_cost(df_elements),
    }
    reg.append(entry)
    _save_registry(reg)
    return proj_id


def list_projects() -> pd.DataFrame:
    """Return a summary DataFrame of all saved projects."""
    reg = _load_registry()
    if not reg:
        return pd.DataFrame()
    rows = []
    for p in reg:
        pid = p.get('id') or p.get('project_id', '')
        rows.append({
            'ID':            pid,
            'Name':          p['name'],
            'Type':          p['building_type'],
            'City':          p.get('city', ''),
            'Floors':        p['floors'],
            'Floor Area (m²)': p['floor_area_m2'],
            'Duration (d)':  p['duration_days'],
            'Elements':      p['n_total'],
            'Clusters':      p.get('n_clusters', '—'),
            'Saved':         p['saved_at'][:10],
        })
    return pd.DataFrame(rows)


def load_project_df(project_id: str) -> pd.DataFrame:
    """Load the structural_elements CSV for a saved project."""
    reg = _load_registry()
    for p in reg:
        pid = p.get('id') or p.get('project_id', '')
        if pid == project_id:
            return pd.read_csv(p['csv_path'])
    raise KeyError(f"Project {project_id!r} not found in library.")


def delete_project(project_id: str):
    """Remove a project from the library."""
    reg = _load_registry()
    new_reg = []
    for p in reg:
        pid = p.get('id') or p.get('project_id', '')
        if pid == project_id:
            proj_dir = os.path.join(LIBRARY_DIR, project_id)
            if os.path.isdir(proj_dir):
                shutil.rmtree(proj_dir)
        else:
            new_reg.append(p)
    _save_registry(new_reg)


def get_training_dataframe() -> pd.DataFrame:
    """
    Returns a flat DataFrame of all projects suitable for ML training.
    Each row = one project's aggregated stats.
    """
    reg = _load_registry()
    rows = []
    for p in reg:
        rows.append({
            # Input features
            'building_type':  p['building_type'],
            'floors':         p['floors'],
            'floor_area_m2':  p['floor_area_m2'],
            'duration_days':  p['duration_days'],
            # Targets — total counts
            'n_columns':      p['n_columns'],
            'n_slabs':        p['n_slabs'],
            'n_beams':        p['n_beams'],
            'n_total':        p['n_total'],
            'n_clusters':     p.get('n_clusters', p['n_columns'] + p['n_slabs'] + p['n_beams']),
            # Targets — avg dimensions
            'col_length':     p['col_dims']['length'],
            'col_width':      p['col_dims']['width'],
            'col_height':     p['col_dims']['height'],
            'slab_length':    p['slab_dims']['length'],
            'slab_width':     p['slab_dims']['width'],
            'slab_height':    p['slab_dims']['height'],
            'beam_length':    p['beam_dims']['length'],
            'beam_width':     p['beam_dims']['width'],
            'beam_height':    p['beam_dims']['height'],
            # Cost
            'total_cost':     p['total_cost'],
        })
    return pd.DataFrame(rows)
