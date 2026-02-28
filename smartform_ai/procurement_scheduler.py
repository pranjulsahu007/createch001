"""
procurement_scheduler.py
Feature 4 — Procurement Lead Time

Calculates the latest date a purchase order must be placed for each cluster
so that formwork sets arrive before the first pour.
"""
import pandas as pd
from datetime import date


def generate_procurement_schedule(
    df_optimization: pd.DataFrame,
    df_elements: pd.DataFrame,
    lead_time_days: int = 5
) -> pd.DataFrame:
    """
    For every (Cluster_ID, Zone) pair, compute:
      - first pour date
      - order-by date  (first_pour − lead_time_days)
      - procurement status  (URGENT / ON TRACK / PLANNED)

    Parameters
    ----------
    df_optimization : pd.DataFrame
        Output of optimize_formwork_sets(). Must contain:
        Cluster_ID, Zone, Required_Sets, Optimized_Procurement_Cost
    df_elements : pd.DataFrame
        Clustered element data. Must contain:
        Cluster_ID, Casting_Date, Formwork_Cost_per_Set
        Optionally: Zone (Feature 2)
    lead_time_days : int
        Days from placing order to delivery on site.

    Returns
    -------
    pd.DataFrame with columns:
        Cluster_ID, Zone, Sets_To_Order, First_Pour_Date,
        Order_By_Date, Estimated_Cost, Status, Days_Until_Order
    """
    df = df_elements.copy()
    df['Casting_Date'] = pd.to_datetime(df['Casting_Date'])

    if '_zone' not in df.columns:
        df['_zone'] = df['Zone'] if 'Zone' in df.columns else 'All'

    today = pd.Timestamp(date.today())
    records = []

    for _, opt_row in df_optimization.iterrows():
        cluster = opt_row['Cluster_ID']
        zone    = opt_row.get('Zone', 'All')

        # Elements belonging to this (cluster, zone)
        mask = df['Cluster_ID'] == cluster
        if zone != 'All' and '_zone' in df.columns:
            mask = mask & (df['_zone'] == zone)

        subset = df.loc[mask]
        if subset.empty:
            continue

        first_pour   = subset['Casting_Date'].min()
        order_by     = first_pour - pd.Timedelta(days=lead_time_days)
        days_to_order = (order_by - today).days

        # Status classification
        if days_to_order < 0:
            status = '🔴 URGENT'
        elif days_to_order <= lead_time_days:
            status = '🟡 ORDER SOON'
        else:
            status = '🟢 PLANNED'

        records.append({
            'Cluster_ID':      cluster,
            'Zone':            zone,
            'Sets_To_Order':   opt_row['Required_Sets'],
            'First_Pour_Date': first_pour.strftime('%Y-%m-%d'),
            'Order_By_Date':   order_by.strftime('%Y-%m-%d'),
            'Days_Until_Order': days_to_order,
            'Estimated_Cost':  opt_row['Optimized_Procurement_Cost'],
            'Status':          status,
        })

    df_schedule = pd.DataFrame(records)

    # Sort: URGENT first, then by order date
    status_order = {'🔴 URGENT': 0, '🟡 ORDER SOON': 1, '🟢 PLANNED': 2}
    df_schedule['_sort'] = df_schedule['Status'].map(status_order)
    df_schedule = df_schedule.sort_values(['_sort', 'Order_By_Date']).drop(columns='_sort')

    return df_schedule.reset_index(drop=True)
