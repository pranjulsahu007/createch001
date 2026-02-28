import pandas as pd
import matplotlib.pyplot as plt

def simulate_timeline(df_elements, df_optimization, reuse_cycle_days=7):
    """
    Simulates daily usage of optimized formwork inventory over the project lifespan.
    """
    df = df_elements.copy()
    df['Casting_Date'] = pd.to_datetime(df['Casting_Date'])
    
    min_date = df['Casting_Date'].min()
    max_date = df['Casting_Date'].max() + pd.Timedelta(days=reuse_cycle_days)
    all_dates = pd.date_range(min_date, max_date)
    
    opt_dict = df_optimization.set_index('Cluster_ID')['Required_Sets'].to_dict()
    
    timeline = []
    
    for d in all_dates:
        daily_req = 0
        daily_avail = sum(opt_dict.values())
        
        # Elements active today (poured between d-reuse_cycle+1 and d)
        active_mask = (df['Casting_Date'] > d - pd.Timedelta(days=reuse_cycle_days)) & (df['Casting_Date'] <= d)
        daily_req = active_mask.sum()
        
        # Elements poured EXACTLY today
        new_demand_today = (df['Casting_Date'] == d).sum()
        
        # Reused: Total Active minus new demand (approximating reuse of older active sets)
        daily_reused = max(0, daily_req - new_demand_today)
        
        timeline.append({
            'Date': d,
            'Required Sets (Active)': daily_req,
            'Available Sets (Inventory)': daily_avail,
            'Reused Sets': daily_reused
        })
        
    df_timeline = pd.DataFrame(timeline)
    return df_timeline

def plot_inventory_timeline(df_timeline):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df_timeline['Date'], df_timeline['Available Sets (Inventory)'], label='Available Sets (Optimized Inventory)', color='#118DFF', linewidth=2, linestyle='--')
    ax.plot(df_timeline['Date'], df_timeline['Required Sets (Active)'], label='Required Sets (In Use)', color='#0E2A47', linewidth=2)
    ax.fill_between(df_timeline['Date'], 0, df_timeline['Required Sets (Active)'], color='#0E2A47', alpha=0.1)
    
    ax.set_title('Formwork Inventory Timeline', fontsize=14, color='#0E2A47')
    ax.set_xlabel('Project Date', fontsize=12)
    ax.set_ylabel('Number of Formwork Sets', fontsize=12)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
