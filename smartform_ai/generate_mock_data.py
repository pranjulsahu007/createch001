import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def create_mock_data(output_dir="."):
    np.random.seed(42)
    random.seed(42)
    
    types = ['Column', 'Slab', 'Beam']
    start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    
    elements = []
    
    # Generate structural elements
    for i in range(1, 401): # Generating 400 for realistic spread
        t = random.choice(types)
        if t == 'Column':
            l, w, h = random.choice([(0.5, 0.5, 3.0), (0.6, 0.6, 3.0), (0.4, 0.4, 3.0), (0.8, 0.8, 3.5)])
            cost = 1200
        elif t == 'Slab':
            l, w, h = random.choice([(5.0, 5.0, 0.2), (4.0, 4.0, 0.2), (6.0, 6.0, 0.2), (8.0, 5.0, 0.25)])
            cost = 3500
        else:
            l, w, h = random.choice([(5.0, 0.3, 0.5), (4.0, 0.3, 0.5), (6.0, 0.4, 0.6), (7.0, 0.5, 0.7)])
            cost = 1800
            
        floor = random.randint(1, 15)
        # Spread pours heavily over 45 days to create overlapping concurrent demand
        cast_date = start_date + timedelta(days=random.randint(0, 45))
        
        elements.append({
            'Element_ID': f"E-{i:03d}",
            'Type': t,
            'Length': l,
            'Width': w,
            'Height': h,
            'Floor': floor,
            'Casting_Date': cast_date.strftime("%Y-%m-%d"),
            'Formwork_Cost_per_Set': cost
        })
        
    df_elements = pd.DataFrame(elements)
    
    # Ensure realistic scenarios by duplicating some high-recurring elements
    highly_repeated = df_elements.sample(100, replace=True)
    df_elements = pd.concat([df_elements, highly_repeated], ignore_index=True)
    
    # Re-sort Element IDs
    df_elements['Element_ID'] = [f"E-{i:03d}" for i in range(1, len(df_elements) + 1)]
    
    # Save elements
    elements_path = os.path.join(output_dir, 'structural_elements.csv')
    df_elements.to_csv(elements_path, index=False)
    
    # Generate schedule.csv as daily summary mapping
    df_schedule = df_elements.groupby('Casting_Date')['Element_ID'].apply(lambda x: ', '.join(x)).reset_index()
    df_schedule.rename(columns={'Element_ID': 'Elements_Planned', 'Casting_Date': 'Date'}, inplace=True)
    schedule_path = os.path.join(output_dir, 'schedule.csv')
    df_schedule.to_csv(schedule_path, index=False)
    
    print(f"Mock data configured for UI testing at: {output_dir}")

if __name__ == "__main__":
    create_mock_data()
