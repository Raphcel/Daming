import pandas as pd

# Read reference list — only "Fully Tropical"
ref = pd.read_csv("tropical-countries-2026.csv")
fully = ref[ref["tropicalCountries_fullyTropical"] == "Fully Tropical"]

# Map reference names -> FAOSTAT names
name_map = {
    "DR Congo": "Democratic Republic of the Congo",
    "Vietnam": "Viet Nam",
    "Tanzania": "United Republic of Tanzania",
    "Ivory Coast": None,  # will match via fuzzy
    "Republic of the Congo": "Congo",
    "Laos": "Lao People's Democratic Republic",
    "Cape Verde": "Cabo Verde",
    "Micronesia": "Micronesia (Federated States of)",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Cuba": "Cuba",
    "Peru": "Peru",
}

df = pd.read_csv("Merged_Climate_LandCover_Data.csv")

# Build FAOSTAT name set from reference
faostat_names = set()
cote = [a for a in df["Area"].unique() if "ivoire" in a.lower() or "ivoire" in a.lower()]

for _, row in fully.iterrows():
    ref_name = row["country"]
    if ref_name == "Ivory Coast":
        faostat_names.update(cote)
    elif ref_name in name_map:
        faostat_names.add(name_map[ref_name])
    else:
        faostat_names.add(ref_name)

# Also include historical entries present in FAOSTAT
# Ethiopia PDR (historical Ethiopia entry), Sudan (former)
extras = ["Ethiopia PDR", "Sudan (former)"]
for e in extras:
    if e in df["Area"].unique():
        faostat_names.add(e)

print(f"Reference: {len(fully)} fully tropical countries")
print(f"FAOSTAT names to match: {len(faostat_names)}")

df_out = df[df["Area"].isin(faostat_names)]
print(f"\nBefore: {len(df)} rows, {df['Area'].nunique()} areas")
print(f"After:  {len(df_out)} rows, {df_out['Area'].nunique()} areas")

# Check unmatched
matched = set(df_out["Area"].unique())
unmatched = faostat_names - matched
if unmatched:
    print(f"\nWARNING - not found in data: {unmatched}")

print(f"\nKept ({df_out['Area'].nunique()}):")
for c in sorted(df_out["Area"].unique()):
    print(f"  {c}")

df_out.to_csv("Merged_Climate_LandCover_Data_FullyTropical.csv", index=False)
print("\nSaved to Merged_Climate_LandCover_Data_FullyTropical.csv")
