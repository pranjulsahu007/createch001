import pandas as pd
import pulp

def optimize_formwork_sets(df_elements, reuse_cycle_days=7):
    """
    Minimizes the number of formwork sets required per cluster using PuLP.
    Constraints: 
    - Formwork reuse cycle restricts availability.
    - Must satisfy casting schedule (daily demand).
    """
    df = df_elements.copy()
    df['Casting_Date'] = pd.to_datetime(df['Casting_Date'])
    
    # Group by cluster and date to get daily pours
    daily_demand = df.groupby(['Cluster_ID', 'Casting_Date']).size().reset_index(name='Daily_Pours')
    
    results = []
    
    for cluster, group in daily_demand.groupby('Cluster_ID'):
        min_date = group['Casting_Date'].min()
        max_date = group['Casting_Date'].max()
        
        # Create full date series for this cluster's active period
        all_dates = pd.date_range(start=min_date, end=max_date)
        cluster_series = group.set_index('Casting_Date')['Daily_Pours'].reindex(all_dates, fill_value=0)
        
        pours_array = cluster_series.values
        days_count = len(pours_array)
        
        # --- PuLP Optimization ---
        try:
            prob = pulp.LpProblem(f"Minimize_Sets_{cluster}", pulp.LpMinimize)
            
            # Z is the max sets needed concurrently for this cluster
            Z = pulp.LpVariable("Required_Sets", lowBound=0, cat='Integer')
            prob += Z, "Objective_Minimize_Sets"
            
            # Constraints: for each day, the active sets in the reuse window must be <= Z
            for i in range(days_count):
                start_window = max(0, i - reuse_cycle_days + 1)
                active_sum = sum(pours_array[start_window:i+1])
                prob += Z >= active_sum, f"Capacity_Day_{i}"
                
            # Solve LP - try to suppress output by catching print streams if needed, msg=0/False manages most.
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            opt_sets = int(pulp.value(Z)) if pulp.LpStatus[prob.status] == 'Optimal' else 0
        except Exception:
            # Fallback simple iterative approach if PuLP fails
            opt_sets = 0
            for i in range(days_count):
                start_window = max(0, i - reuse_cycle_days + 1)
                active_sum = sum(pours_array[start_window:i+1])
                if active_sum > opt_sets:
                    opt_sets = active_sum
        
        # Naive approach: buy 1 set for each max daily pour * 2, or simply the sum of all pours (worst case no reuse)
        naive_sets = sum(pours_array) 
        
        # Cost Calculation
        cost_per_set = df[df['Cluster_ID'] == cluster]['Formwork_Cost_per_Set'].iloc[0]
        
        opt_cost = opt_sets * cost_per_set
        naive_cost = naive_sets * cost_per_set
        
        results.append({
            'Cluster_ID': cluster,
            'Required_Sets': opt_sets,
            'Naive_Sets': naive_sets,
            'Optimized_Procurement_Cost': opt_cost,
            'Naive_Procurement_Cost': naive_cost,
            'Cost_Savings': naive_cost - opt_cost,
            'Cost_Reduction_%': ((naive_cost - opt_cost) / naive_cost * 100) if naive_cost > 0 else 0
        })
        
    df_results = pd.DataFrame(results)
    return df_results
