import pandas as pd
import numpy as np

input_path = 'final.csv'
output_path = 'updated_final.csv'

df = pd.read_csv(input_path)

q_cols = [c for c in df.columns if c.startswith('Q')]

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

map_bin = {'Yes': 1, 'No': 0}
rev_map = {0: 'No', 1: 'Yes'}
for col in q_cols:
    df[col + '_num'] = df[col].map(map_bin)

for col in q_cols:
    tmp = df[['Year', col + '_num']].dropna().sort_values('Year')
    if len(tmp) < 2:
        continue
    x = tmp['Year'].values
    y = tmp[col + '_num'].values
    mask = df[col].isna() | (df[col] == '')
    years_to_fill = df.loc[mask, 'Year'].values
    interp_vals = np.interp(years_to_fill, x, y).round().astype(int)
    df.loc[mask, col] = [rev_map[val] for val in interp_vals]

df.drop(columns=[c + '_num' for c in q_cols], inplace=True)

df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")