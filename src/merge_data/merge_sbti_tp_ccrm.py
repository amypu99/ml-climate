import pandas as pd
from difflib import SequenceMatcher

def similarity(a, b):
    """Returns a fuzzy-match ratio between two strings (0.0–1.0)."""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# 1) Load both CSVs
final_df = pd.read_csv(
    './merged_sbti_tp.csv',
    parse_dates=['Assessment Date']
)
ccrm_df = pd.read_csv('./CCRM/extracted_CCRM_combined.csv')

# 2) Extract year from the Final file’s Assessment Date
final_df['Assessment Year'] = final_df['Assessment Date'].dt.year

# 3) Prepare the result DataFrame and ensure all CCRM feature columns exist
threshold = 0.75  # 75% similarity cutoff
result_df = final_df.copy()
feature_cols = [c for c in ccrm_df.columns if c not in ['Year', 'Name']]
for col in feature_cols:
    if col not in result_df.columns:
        result_df[col] = ''

# 4) Iterate through each row in the CCRM file
for _, ccrm_row in ccrm_df.iterrows():
    c_name = ccrm_row['Name']
    c_year = int(ccrm_row['Year'])
    
    # Find the best fuzzy match in Final.company_name
    best_match, best_score = None, 0
    for f_name in final_df['company_name'].unique():
        score = similarity(c_name, f_name)
        if score > best_score:
            best_match, best_score = f_name, score
    
    if best_score >= threshold:
        # If matched company + year exists, update that row
        mask = (
            (result_df['company_name'] == best_match) &
            (result_df['Assessment Year'] == c_year)
        )
        if mask.any():
            for col in feature_cols:
                result_df.loc[mask, col] = ccrm_row[col]
        else:
            # Otherwise, append a new row for that year
            new_row = {col: '' for col in result_df.columns}
            new_row['company_name'] = best_match
            for col in feature_cols:
                new_row[col] = ccrm_row[col]
            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # No good match → append as a completely new company row
        new_row = {col: '' for col in result_df.columns}
        new_row['company_name'] = c_name
        for col in feature_cols:
            new_row[col] = ccrm_row[col]
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

# 5) Cleanup and save
result_df.drop(columns=['Assessment Year'], inplace=True)
result_df.to_csv('merged_data.csv', index=False)
