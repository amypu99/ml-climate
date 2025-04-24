import pandas as pd

ccrm = pd.read_csv("./CCRM/extracted_CCRM_combined.csv") 
mq   = pd.read_csv("MQ_Assessments_Methodology_5_26032025.csv")

ccrm["Name_std"] = ccrm["Name"].str.strip().str.upper()
mq["Company Name_std"] = mq["Company Name"].str.strip().str.upper()

mq["Assessment Date"] = pd.to_datetime(mq["Assessment Date"], errors="coerce")
mq["Year"] = mq["Assessment Date"].dt.year

companies = ccrm["Name_std"].unique()
mq_filt = mq[mq["Company Name_std"].isin(companies)]

merged = pd.merge(
    ccrm,
    mq_filt,
    how="outer",
    left_on=["Name_std", "Year"],
    right_on=["Company Name_std", "Year"],
    suffixes=("_CCRM", "_MQ"),
)

merged["Company"] = merged["Name"].fillna(merged["Company Name"])
merged.drop(columns=["Name", "Company Name", "Name_std", "Company Name_std"], inplace=True)

merged["Assessment Date"] = merged["Year"]

cols = ["Company", "Year", "Assessment Date"] + [c for c in merged.columns if c not in ("Company", "Year", "Assessment Date")]
merged = merged[cols]

numeric_cols = merged.select_dtypes(include="number").columns.difference(["Year", "Assessment Date"])
merged[numeric_cols] = (
    merged
        .groupby("Company")[numeric_cols]
        .apply(lambda grp: grp.interpolate(method="linear", limit_direction="both"))
        .reset_index(drop=True)
)

output_path = "merged_ccrm_tp_filtered.csv"
merged.to_csv(output_path, index=False)
print(f"Merged CSV saved to {output_path}")