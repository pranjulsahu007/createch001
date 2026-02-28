import pandas as pd
import matplotlib.pyplot as plt

def detect_repetitions(df_elements):
    """
    Groups elements by Type + Dimensions to find repeating formwork sets.
    """
    df = df_elements.copy()
    
    # Group by structural properties
    group_cols = ['Type', 'Length', 'Width', 'Height']
    cluster_summary = df.groupby(group_cols).size().reset_index(name='Frequency')
    
    # Sort by most frequent
    cluster_summary = cluster_summary.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    # Assign Cluster IDs
    cluster_summary['Cluster_ID'] = [f"CL-{(i+1):03d}" for i in range(len(cluster_summary))]
    
    # Merge back to assign Cluster_ID to every element
    df = df.merge(cluster_summary[group_cols + ['Cluster_ID']], on=group_cols, how='left')
    
    return df, cluster_summary

def plot_repetition_bar_chart(cluster_summary):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Limit to top 10 clusters for cleaner visualization
    top_clusters = cluster_summary.head(10)
    
    ax.bar(top_clusters['Cluster_ID'], top_clusters['Frequency'], color='#0E2A47')
    ax.set_title('Top 10 Most Frequent Structural Elements', fontsize=14, color='#0E2A47')
    ax.set_xlabel('Cluster ID (Type + Dimensions)', fontsize=12)
    ax.set_ylabel('Repetition Count', fontsize=12)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
