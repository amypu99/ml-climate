import pandas as pd
import numpy as np

# Paths
merged_path = 'merged_ccrm_tp_filtered.csv'
mq_path     = './raw_label_data/TPI sector data - All sectors - 26032025/MQ_Assessments_Methodology_5_26032025.csv'
output_path = 'final.csv'

df_merged = pd.read_csv(merged_path)
mq        = pd.read_csv(mq_path)

new_companies = [
    "Applied Materials","ON Semiconductor","Wayfair","Chewy","Microsoft","Oracle",
    "Salesforce","Adobe","Intuit","ServiceNow","Workday","Apple","Dell Technologies","HP",
    "Hewlett Packard Enterprise","Western Digital","Super Micro Computer","Nvidia","Intel",
    "Qualcomm","Broadcom","Jabil","Micron Technology","Advanced Micro Devices","Texas Instruments",
    "Lam Research","Analog Devices","KLA","Sanmina","Microchip Technology","Cisco Systems",
    "Amphenol","Motorola Solutions","IBM","CDW","Kyndryl Holdings","Leidos Holdings","DXC Technology",
    "Booz Allen Hamilton Holding","Booz Allen Hamilton Holding (GHG emissions)","Insight Enterprises",
    "Science Applications International","Concentrix","Amazon","Alphabet (Google)","Meta Platforms",
    "Uber Technologies","Coupang","Booking Holdings","Expedia Group","Qurate Retail","eBay","Airbnb",
    "DoorDash","Walmart","Procter & Gamble","Energy Transfer","Cardinal Health","Chevron","Ford Motor",
    "Bank of America","General Motors","Elevance Health","American Airlines Group",
    "Liberty Mutual Insurance Group","UnitedHealth Group","Berkshire Hathaway","CVS Health","Exxon Mobil",
    "McKesson","Cencora","Costco","JPMorgan Chase","Cigna Group","Citigroup","Centene","Home Depot",
    "Marathon Petroleum","Kroger","Phillips 66","Fannie Mae","Walgreens Boots Alliance","Valero Energy",
    "Verizon Communications","AT&T","Comcast","Wells Fargo","Goldman Sachs Group","Freddie Mac","Target",
    "Humana","State Farm Insurance","Tesla","Morgan Stanley","Johnson & Johnson","Archer Daniels Midland",
    "PepsiCo","United Parcel Service","FedEx","Walt Disney","Lowe's","Boeing","Albertsons","Sysco","RTX",
    "General Electric","Lockheed Martin","American Express","Caterpillar","MetLife","HCA Healthcare",
    "Progressive","Deere","StoneX Group","Merck","ConocoPhillips","Pfizer","Delta Air Lines","TD Synnex",
    "Publix Super Markets","Allstate","Nationwide","Charter Communications","AbbVie","New York Life Insurance",
    "TJX","Prudential Financial","Performance Food Group","Tyson Foods","Nike","Enterprise Products Partners",
    "Capital One Financial","Plains GP Holdings","World Kinect","American International Group","Coca-Cola",
    "TIAA","CHS","Bristol-Myers Squibb","Dow","Best Buy","Thermo Fisher Scientific",
    "Massachusetts Mutual Life Insurance","United Services Automobile Assn.","General Dynamics","Travelers",
    "Warner Bros. Discovery","U.S. Bancorp","Abbott Laboratories","Northrop Grumman","Northwestern Mutual",
    "Dollar General"
]

existing = set(df_merged['Company'])
to_add   = [c for c in new_companies if c not in existing]

base_cols = [
    'Geography','Geography Code','Sector','CA100 Company?','Large/Medium Classification',
    'Performance compared to previous year'
]
q_cols = [c for c in mq.columns if c.startswith('Q')]
transfer_cols = base_cols + q_cols

new_rows = []
for comp in to_add:
    subset = mq[mq['Company Name'] == comp]
    if not subset.empty:
        for _, mg in subset.iterrows():
            row = {c: '' for c in df_merged.columns}
            row['Company'] = comp
            try:
                row['Year'] = pd.to_datetime(mg['Assessment Date']).year
            except:
                row['Year'] = ''
            for c in transfer_cols:
                row[c] = mg.get(c, '')
            new_rows.append(row)
    else:
        row = {c: '' for c in df_merged.columns}
        row['Company'] = comp
        row['Year']    = ''
        new_rows.append(row)

df_new = pd.DataFrame(new_rows)

df_all = pd.concat([df_merged, df_new], ignore_index=True)

score_cols = [
    "Overall transparency score","Overall integrity score",
    "1. Transparency and Integrity","2. Transparency","2. Integrity",
    "3. Transparency","3. Integrity","4. Transparency","4. Integrity"
]

ord_map     = {'Very Poor':1, 'Poor':2, 'Moderate':3, 'Reasonable':4, 'High':5}
reverse_map = {v:k for k,v in ord_map.items()}

orig_len = len(df_merged)

for col in score_cols:
    # build numeric x/y from the originals
    yrs = pd.to_numeric(df_merged['Year'], errors='coerce')
    vals = df_merged[col].map(ord_map)
    mask = (~yrs.isna()) & (~vals.isna())
    x = yrs[mask].astype(float).values
    y = vals[mask].astype(float).values
    if len(x) < 2:
        continue
    order = np.argsort(x)
    x, y = x[order], y[order]
    for idx in range(orig_len, len(df_all)):
        yr = df_all.at[idx, 'Year']
        if pd.notna(yr) and yr != '':
            num = np.interp(float(yr), x, y)
            df_all.at[idx, col] = num

def to_cat(val):
    try:
        n = int(round(float(val)))
        n = max(1, min(5, n))
        return reverse_map[n]
    except:
        return val

for col in score_cols:
    df_all[col] = df_all[col].apply(to_cat)

df_all.to_csv(output_path, index=False)
