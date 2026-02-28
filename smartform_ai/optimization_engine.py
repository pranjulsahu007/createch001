import pandas as pd
import math
import pulp

def optimize_formwork_sets(
    df_elements,
    reuse_cycle_days=7,
    max_reuse_count=-1,          # Feature 1: -1 = unlimited
    zone_col=None                 # Feature 2: None = no zone splitting
):
    """
    Minimises the number of formwork sets required per cluster (and optionally
    per zone) using PuLP Integer Linear Programming.

    Parameters
    ----------
    df_elements      : pd.DataFrame  Clustered element data.
    reuse_cycle_days : int           Days a set is locked after a pour (curing + stripping + handling).
    max_reuse_count  : int           Feature 1 — Max times a set can be reused before write-off.
                                     -1 = unlimited (original behaviour).
    zone_col         : str | None    Feature 2 — Column name holding zone labels.
                                     None = treat entire project as one zone.

    Returns
    -------
    pd.DataFrame with one row per (Cluster, Zone) containing:
        Cluster_ID, Zone, Required_Sets, Naive_Sets,
        Optimized_Procurement_Cost, Naive_Procurement_Cost,
        Cost_Savings, Cost_Reduction_%,
        Sets_Written_Off, Write_Off_Cost, True_Total_Cost   (Feature 1)
    """
    df = df_elements.copy()
    df['Casting_Date'] = pd.to_datetime(df.get('Casting_Date', pd.NaT))

    # ── Feature 2: build the grouping key ────────────────────────────────────
    if zone_col and zone_col in df.columns:
        df['_zone'] = df[zone_col].fillna('Default').astype(str)
    else:
        df['_zone'] = 'All'

    group_keys = ['Cluster_ID', '_zone']

    # Daily demand per (cluster, zone, date)
    daily_demand = (
        df.groupby(group_keys + ['Casting_Date'])
          .size()
          .reset_index(name='Daily_Pours')
    )

    results = []

    for (cluster, zone), group in daily_demand.groupby(group_keys):
        min_date = group['Casting_Date'].min()
        max_date = group['Casting_Date'].max()

        all_dates = pd.date_range(start=min_date, end=max_date)
        cluster_series = (
            group.set_index('Casting_Date')['Daily_Pours']
                 .reindex(all_dates, fill_value=0)
        )
        pours_array = cluster_series.values
        days_count  = len(pours_array)
        total_pours = int(pours_array.sum())

        # ── Feature 1: life-limit lower bound ────────────────────────────────
        # If max_reuse_count is set, we need at least ceil(total_pours / max_reuse_count) sets
        life_limit_lb = (
            math.ceil(total_pours / max_reuse_count)
            if max_reuse_count > 0 else 0
        )

        # ── PuLP LP ──────────────────────────────────────────────────────────
        try:
            prob = pulp.LpProblem(f"Sets_{cluster}_{zone}", pulp.LpMinimize)
            Z = pulp.LpVariable("Required_Sets", lowBound=max(0, life_limit_lb), cat='Integer')
            prob += Z

            for i in range(days_count):
                window_start = max(0, i - reuse_cycle_days + 1)
                active_sum   = int(sum(pours_array[window_start:i + 1]))
                prob += Z >= active_sum, f"Cap_Day_{i}"

            # Feature 1 explicit constraint (also enforced via lowBound, belt-and-braces)
            if life_limit_lb > 0:
                prob += Z >= life_limit_lb, "Life_Limit_LB"

            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            opt_sets = (
                int(pulp.value(Z))
                if pulp.LpStatus[prob.status] == 'Optimal' else life_limit_lb
            )
        except Exception:
            # Pure-Python sliding-window fallback
            opt_sets = life_limit_lb
            for i in range(days_count):
                window_start = max(0, i - reuse_cycle_days + 1)
                opt_sets = max(opt_sets, int(sum(pours_array[window_start:i + 1])))

        naive_sets = total_pours  # worst case: buy new set for every pour

        # Cost per set for this (cluster, zone) slice
        mask = (df['Cluster_ID'] == cluster) & (df['_zone'] == zone)
        cost_per_set = df.loc[mask, 'Formwork_Cost_per_Set'].iloc[0]

        # ── Feature 1: write-off cost ─────────────────────────────────────────
        if max_reuse_count > 0:
            sets_written_off = math.floor(total_pours / max_reuse_count)
        else:
            sets_written_off = 0

        replacement_cost = df.loc[mask, 'Replacement_Cost_per_Set'].iloc[0] \
            if 'Replacement_Cost_per_Set' in df.columns else cost_per_set
        write_off_cost = sets_written_off * replacement_cost

        opt_cost   = opt_sets   * cost_per_set
        naive_cost = naive_sets * cost_per_set

        results.append({
            'Cluster_ID':                  cluster,
            'Zone':                        zone,
            'Required_Sets':               opt_sets,
            'Naive_Sets':                  naive_sets,
            'Optimized_Procurement_Cost':  opt_cost,
            'Naive_Procurement_Cost':      naive_cost,
            'Cost_Savings':                naive_cost - opt_cost,
            'Cost_Reduction_%':            ((naive_cost - opt_cost) / naive_cost * 100) if naive_cost > 0 else 0,
            # Feature 1 extras
            'Sets_Written_Off':            sets_written_off,
            'Write_Off_Cost':              write_off_cost,
            'True_Total_Cost':             opt_cost + write_off_cost,
        })

    return pd.DataFrame(results)
