# SmartForm AI — Technical Documentation

> **Version:** 1.0  
> **Stack:** Python 3.10 · Streamlit · Pandas · PuLP · Matplotlib  
> **Entry point:** `app.py`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Input Data Specification](#3-input-data-specification)
4. [Module Reference](#4-module-reference)
   - 4.1 [boq_generator.py](#41-boq_generatorpy)
   - 4.2 [repetition_engine.py](#42-repetition_enginepy)
   - 4.3 [optimization_engine.py](#43-optimization_enginepy)
   - 4.4 [inventory_simulator.py](#44-inventory_simulatorpy)
   - 4.5 [generate_mock_data.py](#45-generate_mock_datapy)
   - 4.6 [app.py](#46-apppy)
5. [Mathematical Models](#5-mathematical-models)
6. [Configuration Reference](#6-configuration-reference)
7. [Output Reference](#7-output-reference)
8. [Running Locally](#8-running-locally)

---

## 1. Project Overview

SmartForm AI is a construction-tech decision-support tool that tackles one of the most expensive and wasteful problems in building construction: **formwork over-procurement**.

Formwork (shuttering) is the temporary mould into which concrete is poured. On a typical multi-floor project, different structural elements — columns, slabs, beams — share identical dimensions across floors. Instead of recognising this and reusing the same formwork set after the concrete has cured (typically 7 days), most site procurement teams buy fresh sets for every element. This leads to:

- **30–70% excess inventory** sitting idle on site
- **Higher logistics and storage costs** (carrying costs)
- **Wasted capital** locked into formwork that is never fully utilised

SmartForm AI solves this by:

1. **Detecting** which elements share dimensions (clusters)
2. **Optimising** via linear programming how many unique formwork sets are truly needed
3. **Simulating** daily inventory utilisation across the project timeline
4. **Generating** a Bill of Quantities (BoQ) with exact formwork areas

---

## 2. Architecture & Data Flow

```
structural_elements.csv
        │
        ▼
┌───────────────────┐
│  boq_generator    │  → adds Formwork_Area_m2 column
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ repetition_engine │  → assigns Cluster_ID to each element
└───────────────────┘
        │
        ▼
┌────────────────────────┐
│  optimization_engine   │  → runs PuLP LP per cluster
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│  inventory_simulator   │  → daily in-use vs available chart
└────────────────────────┘
        │
        ▼
┌───────────────┐
│    app.py     │  → Streamlit dashboard rendering
└───────────────┘
```

Each module is **stateless and pure** — it receives a DataFrame and returns a new DataFrame. No global state is shared between modules.

---

## 3. Input Data Specification

### `structural_elements.csv` *(required)*

| Column | Type | Description |
|---|---|---|
| `Element_ID` | string | Unique identifier, e.g. `E-001` |
| `Type` | string | Must be exactly `Column`, `Slab`, or `Beam` |
| `Length` | float | Length in metres |
| `Width` | float | Width in metres |
| `Height` | float | Height in metres |
| `Floor` | integer | Floor number (informational only) |
| `Casting_Date` | date string | Format: `YYYY-MM-DD` |
| `Formwork_Cost_per_Set` | float | Cost in currency units per single formwork set |

**Minimum rows:** 1. Realistic usage: 50–1000+ rows.

### `schedule.csv` *(optional)*

| Column | Type | Description |
|---|---|---|
| `Date` | date string | Format: `YYYY-MM-DD` |
| `Elements_Planned` | string | Comma-separated Element_IDs planned for that day |

This file is used for schedule preview only. The actual optimisation reads dates directly from `structural_elements.csv`.

---

## 4. Module Reference

---

### 4.1 `boq_generator.py`

**Purpose:** Calculates the total formwork contact area (m²) for every structural element. This area directly drives material quantity estimates and cost rates in real BoQ documents.

#### Function: `generate_boq(df_elements)`

**Accepts:**

| Parameter | Type | Description |
|---|---|---|
| `df_elements` | `pd.DataFrame` | The raw structural elements dataset. Must contain: `Type`, `Length`, `Width`, `Height`. |

**Returns:**

| Field | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Same DataFrame with one new column: `Formwork_Area_m2` (float) |

**Logic:**

The function applies three different geometric formulas based on the `Type` column using Pandas boolean masks (no loops, fully vectorised):

- `Column mask` → row-level formula applied
- `Slab mask` → row-level formula applied
- `Beam mask` → row-level formula applied
- All other types → area remains `0.0`

The original DataFrame is not mutated (`.copy()` is used internally).

**Formula derivation explained:**

| Type | Contact Surfaces | Formula |
|---|---|---|
| Column | 4 vertical faces (all sides) | `2 × (L + W) × H` |
| Slab | 1 bottom face only (soffit) | `L × W` |
| Beam | 2 sides + 1 bottom | `2 × (L + H) × W` |

---

### 4.2 `repetition_engine.py`

**Purpose:** Identifies which structural elements are dimensionally identical (same Type + Length + Width + Height), groups them into clusters, and ranks clusters by how often they repeat. This is the core insight: identical elements can share the same physical formwork set.

#### Function: `detect_repetitions(df_elements)`

**Accepts:**

| Parameter | Type | Description |
|---|---|---|
| `df_elements` | `pd.DataFrame` | Must contain: `Type`, `Length`, `Width`, `Height`. |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Original DataFrame + new column `Cluster_ID` assigned to each element |
| `cluster_summary` | `pd.DataFrame` | One row per unique cluster, sorted by `Frequency` descending |

**`cluster_summary` schema:**

| Column | Description |
|---|---|
| `Type` | Element type |
| `Length` | Dimension |
| `Width` | Dimension |
| `Height` | Dimension |
| `Frequency` | How many elements share this exact dimension set |
| `Cluster_ID` | Assigned label e.g. `CL-001` (highest frequency = lowest number) |

**Logic:**

1. Groups by `['Type', 'Length', 'Width', 'Height']` using `groupby().size()`
2. Sorts descending by frequency
3. Assigns ordered IDs: `CL-001`, `CL-002`, ...
4. Left-merges IDs back onto the original element-level DataFrame

#### Function: `plot_repetition_bar_chart(cluster_summary)`

**Accepts:** `cluster_summary` DataFrame (output of `detect_repetitions`)  
**Returns:** `matplotlib.figure.Figure` — horizontal bar chart of top-10 clusters by frequency

---

### 4.3 `optimization_engine.py`

**Purpose:** Determines the **minimum number of physical formwork sets** that must be procured for each cluster, given that sets cannot be reused until `reuse_cycle_days` have elapsed since the pour.

This is the mathematical core of SmartForm AI.

#### Function: `optimize_formwork_sets(df_elements, reuse_cycle_days=7)`

**Accepts:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df_elements` | `pd.DataFrame` | — | Must contain: `Cluster_ID`, `Casting_Date`, `Formwork_Cost_per_Set` |
| `reuse_cycle_days` | `int` | `7` | Number of days a formwork set is locked before it can be reused (concrete curing time). Configurable in the UI. |

**Returns:**

`pd.DataFrame` with one row per cluster:

| Column | Description |
|---|---|
| `Cluster_ID` | The cluster |
| `Required_Sets` | Minimum sets needed (optimised by LP) |
| `Naive_Sets` | Sets needed if no reuse is planned (worst case — buy one per pour) |
| `Optimized_Procurement_Cost` | `Required_Sets × cost_per_set` |
| `Naive_Procurement_Cost` | `Naive_Sets × cost_per_set` |
| `Cost_Savings` | `Naive_Cost − Optimised_Cost` |
| `Cost_Reduction_%` | Percentage savings |

**How the optimisation works (step by step):**

1. For each cluster, extract all `Casting_Date` values and compute `Daily_Pours` (how many elements of this cluster are cast per day).

2. Build a **continuous daily series** from the cluster's first to last pour date (zero-fills days with no pours). This ensures no gaps in the reuse window calculation.

3. Formulate a **PuLP Integer Linear Programme (ILP)**:
   - **Decision variable:** `Z` — a single integer representing the peak concurrent sets needed
   - **Objective:** `Minimise Z`
   - **Constraints:** For every day `i` in the series, the sum of pours over the window `[i − reuse_cycle + 1, i]` must be ≤ `Z`. This enforces that any set poured in the last N days is still locked (in use / curing) and cannot be reassigned.

4. Solves using the **CBC (Coin-or Branch and Cut) solver** bundled with PuLP (`msg=False` suppresses console output).

5. If PuLP fails for any reason, a pure-Python sliding-window fallback computes the same value iteratively.

6. **Naive baseline** (`Naive_Sets`) = sum of all pours in the cluster (assuming zero reuse — a new set is bought for every pour). This is the baseline that most sites implicitly operate at.

**Key insight:** The LP's per-cluster result is not just an estimate — it is the mathematically provable minimum. Any fewer sets would violate the schedule on at least one day.

---

### 4.4 `inventory_simulator.py`

**Purpose:** Simulates what happens to the formwork inventory every day across the full project lifespan. Shows the gap between what is available (what was optimally procured) and what is actually in use on each day.

#### Function: `simulate_timeline(df_elements, df_optimization, reuse_cycle_days=7)`

**Accepts:**

| Parameter | Type | Description |
|---|---|---|
| `df_elements` | `pd.DataFrame` | Clustered element data with `Casting_Date` |
| `df_optimization` | `pd.DataFrame` | Output of `optimize_formwork_sets` |
| `reuse_cycle_days` | `int` | Reuse cycle (must match the value used in optimisation) |

**Returns:**

`pd.DataFrame` with one row per calendar day from project start to `max_date + reuse_cycle_days`:

| Column | Description |
|---|---|
| `Date` | Calendar date |
| `Required Sets (Active)` | Elements poured within the last N days — i.e. sets currently locked on site |
| `Available Sets (Inventory)` | Total optimised procurement across all clusters (constant — this is what you bought) |
| `Reused Sets` | Active − new pours today (approx. how many sets were reused from a prior day) |

**Logic:**

For each day `d`:
- **Active** = count of elements whose `Casting_Date` falls in `(d − N, d]` — elements poured recently and not yet released
- **Available** = sum of `Required_Sets` across all clusters (the total procured inventory, flat line)
- **Reused** = Active − new pours today (elements active but not freshly poured = reused sets)

The tail end of the timeline (after the last pour) shows sets being released one by one as the final pours cure.

#### Function: `plot_inventory_timeline(df_timeline)`

**Accepts:** `df_timeline` DataFrame (output of `simulate_timeline`)  
**Returns:** `matplotlib.figure.Figure` — dual-line chart showing Available (dashed) vs Active (solid) over the project timeline, with a filled area under the active line.

---

### 4.5 `generate_mock_data.py`

**Purpose:** Creates realistic synthetic datasets so the dashboard can be demonstrated without a real project file. Produces both required CSV files.

#### Function: `create_mock_data(output_dir=".")`

**Accepts:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | `"."` | Directory path where CSVs will be written |

**Produces:**

| File | Description |
|---|---|
| `structural_elements.csv` | 500 rows (400 base + 100 duplicated high-repeat elements to simulate real clustering behaviour) |
| `schedule.csv` | Daily aggregation of which Element_IDs are planned per day |

**Simulation parameters:**

| Element Type | Dimension options (L×W×H) | Cost/Set |
|---|---|---|
| Column | 4 preset sizes (e.g. 0.5×0.5×3.0) | ₹1,200 |
| Slab | 4 preset sizes (e.g. 5×5×0.2) | ₹3,500 |
| Beam | 4 preset sizes (e.g. 5×0.3×0.5) | ₹1,800 |

- Pours are spread randomly over **45 days** from today's date to create overlapping concurrent demand (which is what drives the savings in the LP)
- Seeds are fixed (`np.random.seed(42)`) so output is **reproducible**
- 100 elements are randomly duplicated (with replacement) to boost cluster frequencies and better represent real building repetition patterns

**Run directly from terminal:**
```bash
python generate_mock_data.py
```

---

### 4.6 `app.py`

**Purpose:** The Streamlit web application that wires all modules together and presents results as an interactive dashboard.

**UI Sections:**

| Section | Content |
|---|---|
| Sidebar | Settings sliders (reuse cycle, carrying cost %) + CSV file uploaders |
| Executive KPI Cards | 4 cards: Optimised Cost, Total Savings, Inventory Sets, Carrying Cost Saved |
| 01 — Repetition Detection | Cluster frequency table + horizontal bar chart |
| 02 — Optimisation Results | DataTable with per-cluster LP results + 3 native metric widgets |
| 03 — Inventory Timeline | Line chart of Active vs Available sets over project lifespan |
| Final Summary Banner | Dark-background summary block showing headline numbers |

**Processing pipeline on every load / slider change:**

```
load_data()
    → generate_boq()
    → detect_repetitions()
    → optimize_formwork_sets()   ← cached via @st.cache_data
    → simulate_timeline()
    → render KPIs + charts
```

**Key derived KPIs:**

| KPI | Formula |
|---|---|
| Total Savings | `(Naive_Cost − Opt_Cost) × (1 + carrying_rate)` |
| Excess Reduction % | `(Naive_Sets − Opt_Sets) / Naive_Sets × 100` |
| Reuse Efficiency % | `Σ(Daily_Active) / (Total_Opt_Sets × Project_Days) × 100` |

---

## 5. Mathematical Models

### Formwork Area Formulas

| Element | Formula | Rationale |
|---|---|---|
| Column | `2(L + W) × H` | 4 vertical faces, paired by symmetry |
| Slab | `L × W` | Soffit (underside) form only |
| Beam | `2(L + H) × W` | 2 side faces + 1 soffit, width-scaled |

### Linear Programme (per cluster)

```
Minimise:    Z

Subject to:  Z ≥ Σ pours(t) for t in [day_i − N + 1, day_i],  ∀ day_i
             Z ≥ 0
             Z ∈ ℤ⁺
```

Where:
- `Z` = number of formwork sets to procure
- `N` = reuse cycle in days
- `pours(t)` = number of elements poured on day `t`

The constraint says: on any given day, all pours from the past `N` days are still occupying a set (concrete is still curing). So the peak concurrent demand in any N-day window is the binding constraint.

### Naive Baseline

```
Naive_Sets = Σ all pours in cluster
```

This assumes no reuse occurs at all — one new set is purchased per pour. It represents the worst-case procurement practice.

---

## 6. Configuration Reference

### `.streamlit/config.toml`

```toml
[theme]
base                    = "light"
primaryColor            = "#1565C0"
backgroundColor         = "#F0F2F5"
secondaryBackgroundColor = "#FFFFFF"
textColor               = "#1A1A2E"
font                    = "sans serif"
```

This file must exist for correct rendering. Setting `base = "light"` ensures Streamlit's native widget renderer (dataframes, file uploaders, etc.) uses a light background. Without this file, dark-mode browsers will render widgets with dark backgrounds and invisible text.

### Runtime Parameters (UI Sliders)

| Parameter | Range | Default | Effect |
|---|---|---|---|
| Reuse Cycle (Days) | 3–21 | 7 | Drives the LP window constraint. Longer = fewer reuses possible = more sets required |
| Carrying Cost % | 5–30 | 15 | Applied to optimised savings as `savings × rate` to estimate storage/logistics savings |

---

## 7. Output Reference

### KPI Cards

| Card | Formula | Unit |
|---|---|---|
| Optimised Procurement Cost | `Σ (Required_Sets × Cost_per_Set)` | ₹ |
| Total Savings | `(Naive − Opt) × (1 + carrying_rate)` | ₹ |
| Optimised Inventory | `Σ Required_Sets` | sets |
| Carrying Cost Saved | `(Naive_Cost − Opt_Cost) × carrying_rate` | ₹ |

### Optimisation Table Columns

| Column | Description |
|---|---|
| Cluster | Cluster ID |
| Optimised Sets | LP-derived minimum |
| Naive Sets | No-reuse worst case |
| Optimised Cost (₹) | Sets × cost |
| Savings (₹) | Naive − Optimised cost |
| Reduction % | Savings as % of naive |

---

## 8. Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate demo data (first time only)
python generate_mock_data.py

# 3. Launch dashboard
python -m streamlit run app.py
```

> **Note:** Use `python -m streamlit run app.py` on Windows if `streamlit` is not in your system PATH. The `-m` flag runs the module directly through the Python interpreter.

The app will open automatically at `http://localhost:8501`.

**File structure:**
```
smartform_ai/
├── app.py                    ← Streamlit entry point
├── boq_generator.py          ← Formwork area calculations
├── repetition_engine.py      ← Cluster detection
├── optimization_engine.py    ← PuLP LP solver
├── inventory_simulator.py    ← Daily timeline simulation
├── generate_mock_data.py     ← Demo data generator
├── requirements.txt          ← Python dependencies
├── structural_elements.csv   ← Generated/uploaded input
├── schedule.csv              ← Generated/uploaded input
├── README.md                 ← Quick-start guide
└── .streamlit/
    └── config.toml           ← Theme configuration
```
