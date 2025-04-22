
import pandas as pd
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Load China Shock import exposure data (SIC87-based)
china_exposure_path = "/Users/anthonyperti/Downloads/cw_hs6_sic87dd/cw_hs6_sic87dd.dta"
exposure_df = pd.read_stata(china_exposure_path)

# Load NAICS97-to-SIC87 crosswalk file (uploaded)
crosswalk_path = "/Users/anthonyperti/Downloads/cw_n97_s87/cw_n97_s87.dta"
crosswalk_df = pd.read_stata(crosswalk_path)

# Clean and convert codes
crosswalk_df["NAICS"] = crosswalk_df["naics6"].astype(int).astype(str).str.zfill(6)
crosswalk_df["SIC"] = crosswalk_df["sic4"].astype(int).astype(str).str.zfill(4)
crosswalk_df = crosswalk_df[["NAICS", "SIC"]]  # Drop weight for now

# Preview and prepare exposure data
print("China Shock exposure data columns:", exposure_df.columns)
exposure_df = exposure_df.rename(columns={"sic87dd": "SIC"})
exposure_df = exposure_df[exposure_df["SIC"].notna()]
exposure_df["SIC"] = exposure_df["SIC"].astype(float).astype(int).astype(str).str.zfill(4)

# Load the file
file_path = "/Users/anthonyperti/Downloads/cbp00msa.txt"

# Define columns from CBP MSA layout
columns = [
    "MSA", "NAICS", "EMPFLAG", "EMP", "QP1", "AP", "EST",
    "N1_4", "N5_9", "N10_19", "N20_49", "N50_99",
    "N100_249", "N250_499", "N500_999", "N1000"
]

# Read the file
df = pd.read_csv(file_path, header=None, names=columns, dtype=str)

# Filter for Philadelphia MSA (historic 2000 code is 6160)
philly_df = df[df["MSA"] == "6160"].copy()

# Convert numeric columns to proper types
numeric_cols = ["EMP", "QP1", "AP", "EST", "N1_4", "N5_9", "N10_19", "N20_49",
                "N50_99", "N100_249", "N250_499", "N500_999", "N1000"]
philly_df[numeric_cols] = philly_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Filter for manufacturing industries: NAICS codes starting with 31, 32, or 33
philly_mfg = philly_df[philly_df["NAICS"].str.startswith(("31", "32", "33"), na=False)].copy()

 # Remove duplicate NAICS entries by summing employment
philly_mfg = philly_mfg.groupby("NAICS", as_index=False)["EMP"].sum()

# Ensure EMP is float for calculations
philly_mfg["EMP"] = philly_mfg["EMP"].astype(float)

# Drop any NAICS codes that are not valid 6-digit numbers
philly_mfg = philly_mfg[philly_mfg["NAICS"].str.match(r'^\d{6}$', na=False)].copy()
philly_mfg = philly_mfg[philly_mfg["EMP"] > 0]

# Recalculate total employment and employment share
total_mfg_emp = philly_mfg["EMP"].sum()
philly_mfg["Emp_Share"] = philly_mfg["EMP"] / total_mfg_emp

# Save a copy for later use (for job gains visualization)
philly_mfg_2000 = philly_mfg.copy()

# Save the result
philly_mfg.to_csv("philly_manufacturing_2000.csv", index=False)

# Preview top 5 rows
print(philly_mfg[["NAICS", "EMP", "Emp_Share"]].head())

# Add a 2-digit NAICS column


# Merge in SIC codes using true NAICS-to-SIC crosswalk
philly_mfg = philly_mfg.merge(crosswalk_df, on="NAICS", how="left")

# Ensure SIC codes are 4-digit strings for matching
philly_mfg["SIC"] = philly_mfg["SIC"].astype(str).str.zfill(4)

# Merge with exposure data
merged = philly_mfg.merge(exposure_df, on="SIC", how="left")

# Check for merge success
missing_sic = merged["share"].isna().sum()
total_rows = len(merged)
print(f"Rows missing exposure data: {missing_sic} / {total_rows}")
print("Sample SIC in philly_mfg:", philly_mfg["SIC"].dropna().unique()[:5])
print("Sample SIC in exposure_df:", exposure_df["SIC"].dropna().unique()[:5])

# Print available columns to find the correct exposure variable
print("Available columns in exposure_df:", exposure_df.columns.tolist())

# Use 'share' column from exposure data as the China import exposure measure (normalized)
if "share" in merged.columns:
    # Normalize EMP shares just in case they do not sum to 1 due to merge/missing data
    merged = merged[merged["share"].notna()].copy()
    merged["Emp_Share_Norm"] = merged["EMP"] / merged["EMP"].sum()
    merged["Weighted_Exposure"] = merged["Emp_Share_Norm"] * merged["share"]
    philly_exposure = merged["Weighted_Exposure"].sum()
    print("Corrected Philadelphia China Shock Exposure (normalized):", philly_exposure)
else:
    print("Expected exposure variable 'share' not found in exposure_df.")

philly_mfg["NAICS_2digit"] = philly_mfg["NAICS"].str[:2]

# Group by 2-digit and sum employment
mfg_agg = philly_mfg.groupby("NAICS_2digit")["EMP"].sum().reset_index()
mfg_agg["Emp_Share"] = mfg_agg["EMP"] / mfg_agg["EMP"].sum()

print(mfg_agg)

# Additional context for analysis
print("\nSummary Statistics for Manufacturing Employment in Philadelphia MSA (2000):")
print("Total Manufacturing Employment:", total_mfg_emp)
print("Mean Employment per Industry:", philly_mfg['EMP'].mean())
print("Median Employment per Industry:", philly_mfg['EMP'].median())
print("Top Industries by Employment Share:")
print(philly_mfg.sort_values("Emp_Share", ascending=False)[["NAICS", "EMP", "Emp_Share"]].head(5))



# --- Aggregated Exposure for NAICS 33 ---

df_33 = merged[merged["NAICS"].str.startswith("33", na=False)].copy()
df_33 = df_33[df_33["share"].notna() & df_33["EMP"] > 0].copy()
df_33["EMP"] = df_33["EMP"].astype(float)
df_33["Weighted_Exposure"] = df_33["EMP"] * df_33["share"]

# Aggregate by NAICS to remove duplicates and calculate average exposure
df_33_agg = df_33.groupby("NAICS").agg(
    Total_EMP=("EMP", "sum"),
    Avg_Share=("share", "mean"),
    Weighted_Exposure=("Weighted_Exposure", "sum")
).reset_index()
df_33_agg["Exposure_per_Emp"] = df_33_agg["Weighted_Exposure"] / df_33_agg["Total_EMP"]

# Output aggregated NAICS 33 exposure details
print("\nAggregated Exposure for NAICS 33:")
print(df_33_agg.head(10))
df_33_agg.to_csv("naics33_exposure_details.csv", index=False)


# --- 1997 Manufacturing Employment Analysis ---

# Load 1997 CBP file with proper header
cbp_1997_path = "/Users/anthonyperti/Downloads/cbp97msa.txt"
columns_1997 = [
    "MSA", "SIC", "EMPFLAG", "EMP", "QP1", "AP", "EST",
    "N1_4", "N5_9", "N10_19", "N20_49", "N50_99",
    "N100_249", "N250_499", "N500_999", "N1000"
]
df_1997 = pd.read_csv(cbp_1997_path, header=None, names=columns_1997, dtype=str)

# Filter for Philadelphia MSA using 1997 code (6160)
philly_df_1997 = df_1997[df_1997["MSA"] == "6160"].copy()

# Convert numeric columns to proper types
numeric_cols_1997 = ["EMP", "QP1", "AP", "EST", "N1_4", "N5_9", "N10_19", "N20_49",
                     "N50_99", "N100_249", "N250_499", "N500_999", "N1000"]
philly_df_1997[numeric_cols_1997] = philly_df_1997[numeric_cols_1997].apply(pd.to_numeric, errors='coerce')

# Filter for manufacturing industries: SIC codes starting with 20 or 30-39
philly_mfg_1997 = philly_df_1997[philly_df_1997["SIC"].str.match(r"^(2[0-9]{2}|3[0-9]{2})", na=False)].copy()

# Calculate total manufacturing employment for 1997
total_mfg_emp_1997 = philly_mfg_1997["EMP"].sum()
print("Total Manufacturing Employment in Philadelphia MSA (1997):", total_mfg_emp_1997)


# --- 2007 Manufacturing Employment Analysis ---

# Load 2007 CBP file with proper header
cbp_2007_path = "/Users/anthonyperti/Downloads/cbp07msa.txt"
df_2007 = pd.read_csv(cbp_2007_path, dtype=str)
df_2007.columns = df_2007.columns.str.lower()

# Filter for Philadelphia MSA using 2007 code (37980)
philly_df_2007 = df_2007[df_2007["msa"] == "37980"].copy()
philly_df_2007["emp"] = pd.to_numeric(philly_df_2007["emp"], errors="coerce")

# Filter for manufacturing NAICS codes that start with 31, 32, or 33 and are valid 6-digit codes
philly_mfg_2007 = philly_df_2007[
    philly_df_2007["naics"].str.match(r"^(31|32|33)\d{4}$", na=False)
].copy()

# Calculate and report total manufacturing employment
total_mfg_emp_2007 = philly_mfg_2007["emp"].sum()
print("Total Manufacturing Employment in Philadelphia MSA (2007):", total_mfg_emp_2007)

# Compare with 2000
job_loss = total_mfg_emp - total_mfg_emp_2007
job_loss_pct = (job_loss / total_mfg_emp) * 100

print(f"Job loss from 2000 to 2007: {job_loss} jobs ({job_loss_pct:.2f}%)")

# --- 1997 Manufacturing Employment Analysis ---

# Load 1997 CBP file with proper header
cbp_1997_path = "/Users/anthonyperti/Downloads/cbp97msa.txt"
columns_1997 = [
    "MSA", "SIC", "EMPFLAG", "EMP", "QP1", "AP", "EST",
    "N1_4", "N5_9", "N10_19", "N20_49", "N50_99",
    "N100_249", "N250_499", "N500_999", "N1000"
]
df_1997 = pd.read_csv(cbp_1997_path, header=None, names=columns_1997, dtype=str)

# Filter for Philadelphia MSA using 1997 code (6160)
philly_df_1997 = df_1997[df_1997["MSA"] == "6160"].copy()

# Convert numeric columns to proper types
numeric_cols_1997 = ["EMP", "QP1", "AP", "EST", "N1_4", "N5_9", "N10_19", "N20_49",
                     "N50_99", "N100_249", "N250_499", "N500_999", "N1000"]
philly_df_1997[numeric_cols_1997] = philly_df_1997[numeric_cols_1997].apply(pd.to_numeric, errors='coerce')

# Filter for manufacturing industries: SIC codes starting with 20-39
philly_mfg_1997 = philly_df_1997[philly_df_1997["SIC"].str.match(r"^2[0-9]{2}|3[0-9]{2}", na=False)].copy()

# Calculate total manufacturing employment
total_mfg_emp_1997 = philly_mfg_1997["EMP"].sum()
print("Total Manufacturing Employment in Philadelphia MSA (1997):", total_mfg_emp_1997)

# Compare 1997 to 2000
job_loss_97_00 = total_mfg_emp_1997 - total_mfg_emp
job_loss_pct_97_00 = (job_loss_97_00 / total_mfg_emp_1997) * 100

print(f"Job loss from 1997 to 2000: {job_loss_97_00} jobs ({job_loss_pct_97_00:.2f}%)")


# --- Visualization of Top Manufacturing Job Gains from 2000 to 2007 ---

import matplotlib.pyplot as plt

# Clean and prepare NAICS codes for comparison
philly_mfg_2000["NAICS_clean"] = philly_mfg_2000["NAICS"].str.replace(r"[^\d]", "", regex=True)
philly_mfg_2007["NAICS_clean"] = philly_mfg_2007["naics"].str.replace(r"[^\d]", "", regex=True)

# Merge employment data
naics_growth = pd.merge(
    philly_mfg_2000[["NAICS", "NAICS_clean", "EMP"]],
    philly_mfg_2007[["naics", "NAICS_clean", "emp"]],
    left_on="NAICS_clean", right_on="NAICS_clean", how="outer", suffixes=("_2000", "_2007")
)
naics_growth["EMP_2000"] = naics_growth["EMP"].fillna(0)
naics_growth["EMP_2007"] = naics_growth["emp"].fillna(0)
naics_growth["EMP_Change"] = naics_growth["EMP_2007"] - naics_growth["EMP_2000"]
naics_growth["NAICS"] = naics_growth["NAICS_clean"]

# Map NAICS codes to industry descriptions
naics_descriptions = {
    "32311": "Printing",
    "33451": "Instruments Manufacturing",
    "32619": "Plastics Products",
    "31161": "Animal Processing",
    "33232": "Metal Products",
    "33721": "Office Furniture",
    "33441": "Semiconductors",
    "32541": "Pharmaceuticals",
    "339950": "Sign Manufacturing",
    "33995": "Sign Manufacturing",
    "311812": "Commercial Bakeries",
    "311813": "Frozen Cakes and Pastries",
    "332999": "Misc. Fabricated Metals",
    "333120": "Construction Machinery",
    "333911": "Pump and Compressor Mfg",
    "336390": "Other Motor Vehicle Parts"
}
# Add additional NAICS mappings
naics_descriptions.update({
    "339950": "Sign Manufacturing",
    "336322": "Other Motor Vehicle Electrical Equipment",
    "323122": "Prepress Services",
    "331111": "Iron and Steel Mills",
    "325211": "Plastics Material and Resin",
    "325412": "Pharmaceutical Preparation",
    "334516": "Analytical Laboratory Instruments",
    "337110": "Wood Kitchen Cabinet Manufacturing",
    "325199": "All Other Basic Organic Chemical",
    "323113": "Commercial Printing, Screen"
})

# Filter and prepare top job gains
naics_growth["NAICS_Clean"] = naics_growth["NAICS"].str.replace(r"[^\d]", "", regex=True)
top_growth = naics_growth.sort_values("EMP_Change", ascending=False).head(10).copy()
top_growth["Industry"] = top_growth["NAICS_Clean"].map(naics_descriptions).fillna("Other")

print("\nTop NAICS codes in job gains:")
print(top_growth[["NAICS_Clean", "EMP_Change"]])

# Plot top job gains
plt.figure(figsize=(10, 6))
plt.barh(top_growth["Industry"], top_growth["EMP_Change"], color='seagreen')
plt.title("Top Manufacturing Job Gains in Philadelphia (2000–2007)")
plt.xlabel("Jobs Gained")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- 2010 Manufacturing Employment Analysis ---

import pandas as pd

cbp_2010_path = "/Users/anthonyperti/Downloads/cbp10msa.txt"
cbp_2010 = pd.read_csv(cbp_2010_path, dtype=str)
cbp_2010.columns = cbp_2010.columns.str.lower()
cbp_2010["emp"] = pd.to_numeric(cbp_2010["emp"], errors="coerce")

# Filter for Philadelphia MSA using CBSA code (37980) and valid 6-digit manufacturing NAICS
philly_2010 = cbp_2010[
    (cbp_2010["msa"] == "37980") &
    (cbp_2010["naics"].str.match(r"^(31|32|33)\d{4}$", na=False))
].copy()

total_mfg_emp_2010 = philly_2010["emp"].sum()
print("Total Manufacturing Employment in Philadelphia MSA (2010):", total_mfg_emp_2010)

# --- Philadelphia Manufacturing Job Changes ---
print("\n--- Philadelphia Manufacturing Job Changes ---")
print(f"1997 → 2000: {total_mfg_emp - total_mfg_emp_1997:.0f} jobs ({((total_mfg_emp - total_mfg_emp_1997) / total_mfg_emp_1997) * 100:.2f}%)")
print(f"2000 → 2007: {total_mfg_emp_2007 - total_mfg_emp:.0f} jobs ({((total_mfg_emp_2007 - total_mfg_emp) / total_mfg_emp) * 100:.2f}%)")
print(f"2007 → 2010: {total_mfg_emp_2010 - total_mfg_emp_2007:.0f} jobs ({((total_mfg_emp_2010 - total_mfg_emp_2007) / total_mfg_emp_2007) * 100:.2f}%)")

# Optional: Save output for further analysis
philly_2010.to_csv("philly_manufacturing_2010.csv", index=False)

# --- Consistent NAICS-based Employment Change Analysis (6-digit, 31-33 only) ---
# Clean and filter NAICS in each year to ensure valid 6-digit manufacturing codes
def clean_naics(df, code_col, emp_col, pattern=r"^(31|32|33)\d{4}$"):
    df = df[df[code_col].str.match(pattern, na=False)].copy()
    df[code_col] = df[code_col].str.replace(r"[^\d]", "", regex=True)
    df[emp_col] = pd.to_numeric(df[emp_col], errors="coerce")
    return df.groupby(code_col)[emp_col].sum()

emp_97 = clean_naics(philly_mfg_1997.rename(columns={"SIC": "naics", "EMP": "emp"}), "naics", "emp")
emp_00 = clean_naics(philly_mfg.rename(columns={"NAICS": "naics", "EMP": "emp"}), "naics", "emp")
emp_07 = clean_naics(philly_mfg_2007.rename(columns={"naics": "naics", "emp": "emp"}), "naics", "emp")
emp_10 = clean_naics(philly_2010.rename(columns={"naics": "naics", "emp": "emp"}), "naics", "emp")

# Combine all into one DataFrame
all_years = pd.DataFrame({
    "EMP_97": emp_97,
    "EMP_00": emp_00,
    "EMP_07": emp_07,
    "EMP_10": emp_10
}).fillna(0)

# Calculate changes
all_years["Change_97_00"] = all_years["EMP_00"] - all_years["EMP_97"]
all_years["Change_00_07"] = all_years["EMP_07"] - all_years["EMP_00"]
all_years["Change_07_10"] = all_years["EMP_10"] - all_years["EMP_07"]

# Identify top 2 gainers and losers for each period
for period in ["97_00", "00_07", "07_10"]:
    print(f"\n--- Employment Change {period.replace('_', ' → ')} ---")
    
    print("Top 2 Gainers:")
    top_gainers = all_years[f"Change_{period}"].nlargest(2)
    for idx, val in top_gainers.items():
        industry = naics_descriptions.get(idx, 'Other')
        print(f"NAICS: {idx}, Change: {val:.0f}, Industry: {industry}")
    
    print("Top 2 Losers:")
    top_losers = all_years[f"Change_{period}"].nsmallest(2)
    for idx, val in top_losers.items():
        industry = naics_descriptions.get(idx, 'Other')
        print(f"NAICS: {idx}, Change: {val:.0f}, Industry: {industry}")


# --- China Shock Exposure for Top Gainers and Losers ---
print("\n--- China Shock Exposure for Top Gainers and Losers ---")

# Combine top 2 gainers and losers across all periods
top_naics = set()
for period in ["97_00", "00_07", "07_10"]:
    gainers = all_years[f"Change_{period}"].nlargest(2).index
    losers = all_years[f"Change_{period}"].nsmallest(2).index
    top_naics.update(gainers)
    top_naics.update(losers)

top_naics = list(top_naics)
top_naics_df = pd.DataFrame({"NAICS": top_naics})
top_naics_df["NAICS"] = top_naics_df["NAICS"].astype(str).str.zfill(6)

# Merge to get SIC codes
top_naics_df = top_naics_df.merge(crosswalk_df, on="NAICS", how="left")
top_naics_df["SIC"] = top_naics_df["SIC"].astype(str).str.zfill(4)

# Merge with exposure
top_naics_df = top_naics_df.merge(exposure_df[["SIC", "share"]], on="SIC", how="left")


# Add readable labels
top_naics_df["Industry"] = top_naics_df["NAICS"].map(naics_descriptions).fillna("Other")

# Fallback: Replace 'Other' using U.S. Census NAICS API
import requests

def get_naics_title(naics_code):
    base_url = f"https://api.census.gov/data/2017/naics?get=NAICS2017_LABEL&for=naics:{naics_code}"
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            return response.json()[1][0]
    except Exception as e:
        print(f"Failed to retrieve NAICS title for {naics_code}: {e}")
    return "Unknown"

# Replace 'Other' labels using API
mask_other = top_naics_df["Industry"] == "Other"
top_naics_df.loc[mask_other, "Industry"] = top_naics_df.loc[mask_other, "NAICS"].apply(get_naics_title)


#
# Manual mapping for known missing codes: EXTENDED with full list of codes showing as "Other"
naics_descriptions.update({
    "323110": "Commercial Printing (except Screen and Books)",
    "315212": "Men's and Boys' Cut and Sew Apparel Manufacturing",
    "311111": "Dog and Cat Food Manufacturing",
    "311119": "Other Animal Food Manufacturing",
    "312111": "Soft Drink Manufacturing",
    "339991": "Gasket, Packing, and Sealing Device Manufacturing",
    "323122": "Prepress Services",
    "331111": "Iron and Steel Mills and Ferroalloy Manufacturing",
    "325211": "Plastics Material and Resin Manufacturing",
    "334516": "Analytical Laboratory Instrument Manufacturing",
    "337110": "Wood Kitchen Cabinet and Countertop Manufacturing",
    "325199": "All Other Basic Organic Chemical Manufacturing",
    "323113": "Commercial Screen Printing"
})

# Reapply the mapping to Industry
top_naics_df["Industry"] = top_naics_df["NAICS"].map(naics_descriptions).fillna(top_naics_df["Industry"])




# --- Analysis and Visualization: Exposure and Employment Change for Selected Industries ---

# Focus on two example NAICS codes: 325412 (Pharmaceutical Preparation) and 336322 (Other Motor Vehicle Electrical Equipment)
focus_df = top_naics_df[top_naics_df["NAICS"].isin(["325412", "336322"])].copy()
print("\n--- Focus Industries Exposure Data ---")
print(focus_df)

# 1. Summary statistics for exposure
focus_df = top_naics_df[top_naics_df["NAICS"].isin(["325412", "336322"])].copy()
summary_stats = focus_df.groupby("NAICS", as_index=False).agg(
    Industry=("Industry", "first"),
    Mean_Exposure=("share", "mean"),
    Max_Exposure=("share", "max"),
    Min_Exposure=("share", "min"),
    Count=("share", "count")
)

print("\n--- Exposure Summary Statistics ---")
print(summary_stats)

focus_naics = ["325412", "336322"]
# Employment change data for focus industries
focus_emp_change = all_years.loc[all_years.index.isin(focus_naics)].copy()
focus_emp_change = focus_emp_change.reset_index()
focus_emp_change.rename(columns={focus_emp_change.columns[0]: 'NAICS'}, inplace=True)

print("\n--- Employment Change for Focus Industries ---")
print(focus_emp_change[["NAICS", "EMP_97", "EMP_00", "EMP_07", "EMP_10",
                       "Change_97_00", "Change_00_07", "Change_07_10"]])

# Plot employment trends
import matplotlib.pyplot as plt

for _, row in focus_emp_change.iterrows():
    plt.plot(["1997", "2000", "2007", "2010"], 
             [row["EMP_97"], row["EMP_00"], row["EMP_07"], row["EMP_10"]],
             marker='o', label=naics_descriptions.get(row["NAICS"], row["NAICS"]))

plt.title("Employment Trends: Focus Industries")
plt.xlabel("Year")
plt.ylabel("Employment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Focused China Shock Exposure: Pharmaceutical Preparation & Other Motor Vehicle Electrical Equipment ---
focus_df = top_naics_df[top_naics_df["NAICS"].isin(focus_naics)].copy()

print("\n--- Focused China Shock Exposure: Pharmaceutical Preparation & Other Motor Vehicle Electrical Equipment ---")
print(focus_df[["NAICS", "Industry", "SIC", "share"]])

# Save output tables for documentation or visualization
summary_stats.to_csv("focus_industries_exposure_summary.csv", index=False)
focus_emp_change.to_csv("focus_industries_employment_change.csv", index=False)