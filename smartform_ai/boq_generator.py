import pandas as pd

def generate_boq(df_elements):
    """
    Calculates formwork area for each structural element based on formulas:
    - Column: 2 * (Length + Width) * Height
    - Slab: Length * Width
    - Beam: 2 * (Length + Height) * Width
    """
    df = df_elements.copy()
    
    # Initialize column
    df['Formwork_Area_m2'] = 0.0
    
    # Column Formula
    col_mask = df['Type'] == 'Column'
    df.loc[col_mask, 'Formwork_Area_m2'] = 2 * (df.loc[col_mask, 'Length'] + df.loc[col_mask, 'Width']) * df.loc[col_mask, 'Height']
    
    # Slab Formula
    slab_mask = df['Type'] == 'Slab'
    df.loc[slab_mask, 'Formwork_Area_m2'] = df.loc[slab_mask, 'Length'] * df.loc[slab_mask, 'Width']
    
    # Beam Formula
    beam_mask = df['Type'] == 'Beam'
    df.loc[beam_mask, 'Formwork_Area_m2'] = 2 * (df.loc[beam_mask, 'Length'] + df.loc[beam_mask, 'Height']) * df.loc[beam_mask, 'Width']
    
    return df
