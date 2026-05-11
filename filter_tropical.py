import pandas as pd

df = pd.read_csv("Merged_Climate_LandCover_Data.csv")

# Grab exact names for Côte d'Ivoire and Réunion from the data
cote = [a for a in df["Area"].unique() if "ivoire" in a.lower()]
reun = [a for a in df["Area"].unique() if a.lower().endswith("union")]

# Tropical countries: between Tropic of Cancer (23.5°N) and Tropic of Capricorn (23.5°S)
tropical_countries = [
    # Africa
    "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad",
    "Comoros", "Congo",
    "Democratic Republic of the Congo", "Djibouti",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Ethiopia PDR",
    "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
    "Kenya", "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali",
    "Mauritania", "Mauritius", "Mayotte", "Mozambique", "Namibia",
    "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal",
    "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
    "Sudan", "Sudan (former)", "Togo", "Uganda",
    "United Republic of Tanzania", "Zambia", "Zimbabwe",

    # Americas
    "Antigua and Barbuda", "Bahamas", "Barbados", "Belize",
    "Bolivia (Plurinational State of)", "Brazil", "Colombia",
    "Costa Rica", "Cuba", "Dominica", "Dominican Republic",
    "Ecuador", "El Salvador", "French Guiana", "Grenada",
    "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras",
    "Jamaica", "Martinique", "Mexico", "Nicaragua", "Panama",
    "Paraguay", "Peru", "Puerto Rico", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines",
    "Suriname", "Trinidad and Tobago",
    "Venezuela (Bolivarian Republic of)",

    # Asia
    "Bangladesh", "Brunei Darussalam", "Cambodia", "India", "Indonesia",
    "Lao People's Democratic Republic", "Malaysia", "Maldives",
    "Myanmar", "Oman", "Philippines", "Singapore", "Sri Lanka",
    "Thailand", "Timor-Leste", "Viet Nam", "Yemen",

    # Oceania
    "Fiji", "Kiribati", "Marshall Islands",
    "Micronesia (Federated States of)", "Nauru",
    "Palau", "Papua New Guinea", "Samoa", "Solomon Islands",
    "Tonga", "Tuvalu", "Vanuatu",

    # Territories
    "American Samoa", "Anguilla", "Aruba", "British Virgin Islands",
    "Cayman Islands", "Cook Islands", "French Polynesia",
    "Montserrat", "Netherlands Antilles (former)", "New Caledonia",
    "Niue", "Tokelau", "Turks and Caicos Islands",
    "United States Virgin Islands", "Wallis and Futuna Islands",
    "Pitcairn", "Ascension, Saint Helena and Tristan da Cunha",
]

# Add the encoding-sensitive names found from data
tropical_countries.extend(cote)
tropical_countries.extend(reun)

print(f"Before: {len(df)} rows, {df['Area'].nunique()} unique areas")

df_tropical = df[df["Area"].isin(tropical_countries)]
print(f"After:  {len(df_tropical)} rows, {df_tropical['Area'].nunique()} unique areas")

# Check missing
in_list = set(tropical_countries)
in_data = set(df["Area"].unique())
missing = in_list - in_data
if missing:
    print(f"\nNot found in data: {missing}")

# Save
df_tropical.to_csv("Merged_Climate_LandCover_Data.csv", index=False)
print("\nSaved filtered file.")
