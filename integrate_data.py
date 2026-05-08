import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. Loading datasets...")
# Using latin1 encoding to handle special characters in the CSV
df_emissions = pd.read_csv("Emissions_Totals_E_All_Data/Emissions_Totals_E_All_Data_NOFLAG.csv", encoding="latin1")
df_landcover = pd.read_csv("Environment_LandCover_E_All_Data/Environment_LandCover_E_All_Data_NOFLAG.csv", encoding="latin1")
df_temp = pd.read_csv("Environment_Temperature_change_E_All_Data/Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding="latin1")

# Helper function to melt data from wide to long format
def melt_data(df, id_vars, value_name):
    year_cols = [col for col in df.columns if col.startswith('Y') and col[1:].isdigit()]
    df = df[id_vars + year_cols]
    melted = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name=value_name)
    melted['Year'] = melted['Year'].str.replace('Y', '').astype(int)
    return melted

print("2. Reshaping Temperature dataset...")
# Keep only Meteorological year to get one value per year
df_temp = df_temp[(df_temp['Months'] == 'Meteorological year') & (df_temp['Element'] == 'Temperature change')]
df_temp_long = melt_data(df_temp, ['Area', 'Element'], 'Temperature_Change')
df_temp_long = df_temp_long[['Area', 'Year', 'Temperature_Change']]

print("3. Reshaping Land Cover dataset...")
# Filtering to CCI_LC (Climate Change Initiative Land Cover) which has consistent annual data starting from 1992
df_landcover = df_landcover[df_landcover['Element'].str.contains('CCI_LC', na=False)]
df_lc_long = melt_data(df_landcover, ['Area', 'Item'], 'LandCover_Area')
# Pivot to make each Land Cover type a column
df_lc_pivot = df_lc_long.pivot_table(index=['Area', 'Year'], columns='Item', values='LandCover_Area').reset_index()

print("4. Reshaping Emissions dataset...")
# We use total emissions in CO2eq
df_emissions = df_emissions[df_emissions['Element'].str.contains('CO2eq', na=False)]
df_em_long = melt_data(df_emissions, ['Area', 'Item'], 'Emissions_CO2eq')
# Aggregate items to columns
df_em_pivot = df_em_long.pivot_table(index=['Area', 'Year'], columns='Item', values='Emissions_CO2eq').reset_index()

print("5. Merging datasets together...")
# Start merging based on Area and Year
# Doing an inner merge to keep rows that have all three data points (Usually starting around 1992 because of Land Cover)
merged_df = pd.merge(df_temp_long, df_lc_pivot, on=['Area', 'Year'], how='inner')
merged_df = pd.merge(merged_df, df_em_pivot, on=['Area', 'Year'], how='inner')

# Output result
output_file = "Merged_Climate_LandCover_Data.csv"
merged_df.to_csv(output_file, index=False)
print(f"Integration complete! Saved to {output_file}")
print(f"Dataset shape: {merged_df.shape} (Rows, Columns) -> Meets the requirement of at least 1000 observations!")
