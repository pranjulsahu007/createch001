"""
estimator.py
ML-based formwork BoQ estimator — **production version**.

Changes from v1:
  - Training data comes ONLY from real CSV uploads.  No circular estimate→save
    loop.  The model learns actual element distributions, cluster counts, and
    dimension spread from historical structural_elements CSVs.
  - Cost is computed from formwork area (₹ per m²) instead of flat per-set rate.
  - When trained, the estimator reproduces realistic dimension *clusters* —
    not a single averaged dimension per type.

Model: RandomForestRegressor → MultiOutputRegressor
Fallback: AEC construction norms (hardcoded priors)
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_percentage_error
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from project_library import get_training_dataframe, LIBRARY_DIR

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "project_library", "estimator_model.pkl")
CLUSTER_PATH = os.path.join(os.path.dirname(__file__), "project_library", "cluster_profiles.json")


# ── Default AEC norms (fallback when no training data) ────────────────────────
_PRIORS = {
    'Residential': dict(cpf=4,  spf=1,  bpf=6,
                        cl=0.5, cw=0.5, ch=3.0,
                        sl=5.0, sw=5.0, sh=0.2,
                        bl=5.0, bw=0.3, bh=0.5),
    'Commercial':  dict(cpf=6,  spf=1,  bpf=8,
                        cl=0.6, cw=0.6, ch=3.5,
                        sl=6.0, sw=6.0, sh=0.25,
                        bl=6.0, bw=0.4, bh=0.6),
    'Industrial':  dict(cpf=3,  spf=1,  bpf=5,
                        cl=0.8, cw=0.8, ch=5.0,
                        sl=8.0, sw=8.0, sh=0.3,
                        bl=8.0, bw=0.5, bh=0.7),
    'Mixed-Use':   dict(cpf=5,  spf=1,  bpf=7,
                        cl=0.55, cw=0.55, ch=3.2,
                        sl=5.5,  sw=5.5,  sh=0.22,
                        bl=5.5,  bw=0.35, bh=0.55),
    'Parking':     dict(cpf=3,  spf=1,  bpf=4,
                        cl=0.5, cw=0.5, ch=2.8,
                        sl=7.0, sw=7.0, sh=0.22,
                        bl=7.0, bw=0.4, bh=0.5),
}

# ── Formwork area cost rate defaults (₹ per m²) ──────────────────────────────
DEFAULT_RATE_PER_M2 = 350.0    # reasonable Indian market avg for steel/ply forms


# ── Formwork area computation (mirror of boq_generator.py) ────────────────────
def _formwork_area(etype: str, L: float, W: float, H: float) -> float:
    """Compute formwork contact area (m²) for a single element."""
    if etype == 'Column':
        return 2 * (L + W) * H
    elif etype == 'Slab':
        return L * W
    elif etype == 'Beam':
        return 2 * (L + H) * W
    return 0.0


def cost_from_area(etype: str, L: float, W: float, H: float,
                   rate_per_m2: float = DEFAULT_RATE_PER_M2) -> float:
    """₹ cost per element based on actual formwork area."""
    return round(_formwork_area(etype, L, W, H) * rate_per_m2, 0)


# ── ML targets ────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'n_columns', 'n_slabs', 'n_beams',
    'col_length', 'col_width', 'col_height',
    'slab_length', 'slab_width', 'slab_height',
    'beam_length', 'beam_width', 'beam_height',
    'n_clusters',           # NEW: unique cluster count learned from CSV
]


# ── Cluster profile extraction (from real CSVs) ──────────────────────────────

def _extract_cluster_profiles() -> dict:
    """
    Walk every project in the library and extract per-type dimension clusters.
    Returns
        { 'Column': [ {'L':0.5, 'W':0.5, 'H':3.0, 'count':12}, ... ],
          'Slab':   [ ... ],
          'Beam':   [ ... ] }
    Merged across all projects.
    """
    profiles: dict = {'Column': [], 'Slab': [], 'Beam': []}

    if not os.path.exists(LIBRARY_DIR):
        return profiles

    registry_path = os.path.join(LIBRARY_DIR, 'registry.json')
    if not os.path.exists(registry_path):
        return profiles

    with open(registry_path, encoding='utf-8') as f:
        registry = json.load(f)

    for proj in registry:
        csv_path = os.path.join(LIBRARY_DIR, proj['project_id'], 'structural_elements.csv')
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for etype in profiles:
            sub = df[df['Type'] == etype] if 'Type' in df.columns else pd.DataFrame()
            if sub.empty:
                continue
            # Group by dimension triplet
            dim_cols = [c for c in ['Length', 'Width', 'Height'] if c in sub.columns]
            if len(dim_cols) < 3:
                continue
            for dims, grp in sub.groupby(dim_cols):
                L, W, H = dims
                profiles[etype].append({
                    'L': round(float(L), 3),
                    'W': round(float(W), 3),
                    'H': round(float(H), 3),
                    'count': len(grp),
                })

    # Deduplicate: merge identical L×W×H by summing count
    for etype in profiles:
        merged: dict = {}
        for item in profiles[etype]:
            key = (item['L'], item['W'], item['H'])
            if key in merged:
                merged[key]['count'] += item['count']
            else:
                merged[key] = dict(item)
        profiles[etype] = list(merged.values())
        # Sort by count descending
        profiles[etype].sort(key=lambda x: -x['count'])

    return profiles


# ── Training ──────────────────────────────────────────────────────────────────

def _encode_features(df: pd.DataFrame, le: LabelEncoder = None):
    """Convert building_type to int, return (X, le)."""
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df['btype_enc'] = le.fit_transform(df['building_type'].astype(str))
    else:
        known = set(le.classes_)
        df['building_type'] = df['building_type'].apply(
            lambda x: x if x in known else le.classes_[0])
        df['btype_enc'] = le.transform(df['building_type'].astype(str))

    X = df[['btype_enc', 'floors', 'floor_area_m2', 'duration_days']].values.astype(float)
    return X, le


def train(force: bool = False) -> dict:
    """
    Train the estimator on the project library's **real CSV data**.
    Also extracts cluster profiles for dimension diversity in estimates.
    Returns a dict with model, encoder, n_samples, score, and cluster profiles.
    """
    if not SKLEARN_OK:
        return {'error': 'scikit-learn not installed'}

    df = get_training_dataframe()

    if df.empty or len(df) < 2:
        return {'error': 'not_enough_data', 'n_samples': len(df)}

    # Ensure n_clusters column exists
    if 'n_clusters' not in df.columns:
        df['n_clusters'] = df['n_columns'] + df['n_slabs'] + df['n_beams']

    X, le = _encode_features(df)
    Y     = df[TARGET_COLS].fillna(0).values

    # Choose model based on dataset size
    if len(df) >= 10:
        base = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        base = LinearRegression()

    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    # In-sample MAPE as rough quality indicator
    Y_pred = model.predict(X)
    try:
        score = round(100 - mean_absolute_percentage_error(Y + 1e-6, Y_pred + 1e-6) * 100, 1)
    except Exception:
        score = None

    # Extract cluster profiles from actual CSVs
    cluster_profiles = _extract_cluster_profiles()

    package = {
        'model': model, 'le': le,
        'n_samples': len(df), 'score': score,
        'cluster_profiles': cluster_profiles,
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(package, f)
    # Also save cluster profiles as readable JSON
    with open(CLUSTER_PATH, 'w', encoding='utf-8') as f:
        json.dump(cluster_profiles, f, indent=2)

    return {**package}


def _load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


# ── Estimation ────────────────────────────────────────────────────────────────

def _prior_estimate(building_type, floors, floor_area_m2):
    """Fallback: derive counts from construction norms."""
    p = _PRIORS.get(building_type, _PRIORS['Residential'])
    area_scale = floor_area_m2 / 500.0      # normalised to 500 m² typical floor

    n_col  = max(1, round(p['cpf'] * floors * area_scale))
    n_slab = max(1, round(p['spf'] * floors * area_scale))
    n_beam = max(1, round(p['bpf'] * floors * area_scale))

    return {
        'n_columns': n_col, 'n_slabs': n_slab, 'n_beams': n_beam,
        'col_length':  p['cl'], 'col_width':  p['cw'], 'col_height':  p['ch'],
        'slab_length': p['sl'], 'slab_width': p['sw'], 'slab_height': p['sh'],
        'beam_length': p['bl'], 'beam_width': p['bw'], 'beam_height': p['bh'],
        'n_clusters':  n_col // 2 + n_slab + n_beam // 2,
        'source': 'prior',
        'cluster_profiles': None,
    }


def estimate(
    building_type: str,
    floors: int,
    floor_area_m2: float,
    duration_days: int,
) -> dict:
    """
    Predict element counts + dimensions for a new project.
    Falls back to construction norms if not enough training data.
    """
    if not SKLEARN_OK:
        return _prior_estimate(building_type, floors, floor_area_m2)

    pkg = _load_model()
    if pkg is None or pkg.get('n_samples', 0) < 2:
        result = _prior_estimate(building_type, floors, floor_area_m2)
        result['source'] = 'prior (no model yet — upload real CSVs to library)'
        return result

    X_new = np.array([[0, floors, floor_area_m2, duration_days]], dtype=float)
    known = set(pkg['le'].classes_)
    bt = building_type if building_type in known else pkg['le'].classes_[0]
    X_new[0, 0] = pkg['le'].transform([bt])[0]

    Y_pred = pkg['model'].predict(X_new)[0]
    # Handle old models trained on fewer targets than current TARGET_COLS
    result = {}
    for i, col in enumerate(TARGET_COLS):
        if i < len(Y_pred):
            result[col] = max(0, float(Y_pred[i]))
        else:
            # Fallback for new targets not in old model
            result[col] = 0
    # Round counts to integers
    for k in ('n_columns', 'n_slabs', 'n_beams', 'n_clusters'):
        result[k] = max(1, round(result.get(k, 1)))
    # If n_clusters wasn't predicted, derive from counts
    if result['n_clusters'] <= 1:
        result['n_clusters'] = result['n_columns'] // 2 + result['n_slabs'] + result['n_beams'] // 2
    result['source'] = f"ML model ({pkg['n_samples']} training projects)"
    result['cluster_profiles'] = pkg.get('cluster_profiles')
    return result


# ── Generate elements DataFrame ───────────────────────────────────────────────

def generate_elements_from_estimate(
    est: dict,
    building_type: str,
    floors: int,
    start_date: str,
    duration_days: int,
    zone_names: list = None,
    formwork_rate_per_m2: float = DEFAULT_RATE_PER_M2,
) -> pd.DataFrame:
    """
    Convert an estimate dict into a full structural_elements DataFrame
    that can be fed directly into the existing BoQ / optimisation pipeline.

    If cluster profiles exist (from training on real CSVs), generates elements
    with *varied* dimensions matching the learned distribution.  Otherwise
    uses single averaged dimensions per type.
    """
    if zone_names is None:
        zone_names = ['Zone-A']

    start = datetime.strptime(start_date, '%Y-%m-%d')
    total_elements = est['n_columns'] + est['n_slabs'] + est['n_beams']
    date_range = [start + timedelta(days=int(i * duration_days / max(total_elements - 1, 1)))
                  for i in range(total_elements)]

    profiles = est.get('cluster_profiles') or {}

    rows = []
    idx  = 0
    n_zones = len(zone_names)

    def _distribute_across_profiles(etype, count, dim_key_prefix, fallback_L, fallback_W, fallback_H):
        """
        If cluster profiles exist for etype, distribute `count` elements across
        the known dimension clusters proportionally.  Otherwise, single dimension.
        """
        nonlocal idx
        type_profiles = profiles.get(etype, [])

        if type_profiles:
            # Distribute proportionally to historical counts
            total_hist = sum(p['count'] for p in type_profiles)
            allocated = []
            remaining = count
            for i, p in enumerate(type_profiles):
                if i == len(type_profiles) - 1:
                    n = remaining          # last cluster gets remainder
                else:
                    n = max(1, round(count * p['count'] / total_hist))
                    n = min(n, remaining)
                if n > 0:
                    allocated.append((p['L'], p['W'], p['H'], n))
                    remaining -= n
                if remaining <= 0:
                    break
            # If there are leftover (profiles too few), put remainder on first profile
            if remaining > 0 and allocated:
                l, w, h, c = allocated[0]
                allocated[0] = (l, w, h, c + remaining)
        else:
            # Fallback: single dimension
            allocated = [(fallback_L, fallback_W, fallback_H, count)]

        for L, W, H, n in allocated:
            for i in range(n):
                floor = (idx % floors) + 1
                zone  = zone_names[idx % n_zones]
                area  = _formwork_area(etype, L, W, H)
                cost  = round(area * formwork_rate_per_m2, 0)
                rows.append({
                    'Element_ID':              f"EST-{idx+1:04d}",
                    'Type':                    etype,
                    'Length':                  round(L, 3),
                    'Width':                   round(W, 3),
                    'Height':                  round(H, 3),
                    'Floor':                   floor,
                    'Zone':                    zone,
                    'Casting_Date':            date_range[min(idx, len(date_range)-1)].strftime('%Y-%m-%d'),
                    'Formwork_Area_m2':        round(area, 2),
                    'Formwork_Cost_per_Set':   cost,
                    'Replacement_Cost_per_Set': round(cost * 0.85, 0),
                    'Max_Reuse_Count':         10,
                })
                idx += 1

    _distribute_across_profiles(
        'Column', est['n_columns'],
        'col', est['col_length'], est['col_width'], est['col_height'])
    _distribute_across_profiles(
        'Slab', est['n_slabs'],
        'slab', est['slab_length'], est['slab_width'], est['slab_height'])
    _distribute_across_profiles(
        'Beam', est['n_beams'],
        'beam', est['beam_length'], est['beam_width'], est['beam_height'])

    return pd.DataFrame(rows)
