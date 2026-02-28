import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def create_mock_data(output_dir="."):
    np.random.seed(42)
    random.seed(42)

    types  = ['Column', 'Slab', 'Beam']
    zones  = ['Zone-A', 'Zone-B', 'Zone-C']
    start  = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    elements = []

    for i in range(1, 401):
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

        floor     = random.randint(1, 15)
        cast_date = start + timedelta(days=random.randint(0, 45))
        zone      = random.choice(zones)

        elements.append({
            'Element_ID':             f"E-{i:03d}",
            'Type':                   t,
            'Length':                 l,
            'Width':                  w,
            'Height':                 h,
            'Floor':                  floor,
            'Zone':                   zone,                   # Feature 2
            'Casting_Date':           cast_date.strftime('%Y-%m-%d'),
            'Formwork_Cost_per_Set':  cost,
            'Replacement_Cost_per_Set': round(cost * 0.85, 0),  # Feature 1: replacement usually cheaper
            'Max_Reuse_Count':        10,                     # Feature 1 hint column (informational)
        })

    df = pd.DataFrame(elements)

    # Add 100 duplicated high-repeat rows to simulate real clustering density
    highfreq = df.sample(100, replace=True)
    df = pd.concat([df, highfreq], ignore_index=True)
    df['Element_ID'] = [f"E-{i:03d}" for i in range(1, len(df) + 1)]

    df.to_csv(os.path.join(output_dir, 'structural_elements.csv'), index=False)

    # schedule.csv
    sched = (
        df.groupby('Casting_Date')['Element_ID']
          .apply(lambda x: ', '.join(x))
          .reset_index()
          .rename(columns={'Element_ID': 'Elements_Planned', 'Casting_Date': 'Date'})
    )
    sched.to_csv(os.path.join(output_dir, 'schedule.csv'), index=False)

    print(f"Mock data written to: {output_dir}")
    print(f"  structural_elements.csv — {len(df)} rows, zones: {zones}")

if __name__ == "__main__":
    create_mock_data()
