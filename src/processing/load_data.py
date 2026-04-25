import pandas as pd
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Build path safely
file_path = BASE_DIR / "data" / "raw" / "house_prices.csv"
# here base dir basically replaces all the path from the root to the house-price-predictor folder,
# no need to worry about the exact path on different machines, as long as the structure
# of the project is maintained.

# Load CSV
df = pd.read_csv(file_path)

# Only keep the needed columns 
df = df[["Location", "Date", "CompBenchmark"]]

# rename 
df = df.rename(columns = {
    "Location": "location",
    "Date": "date",
    "CompBenchmark": "price"
})

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Drop rows with missing price
df = df.dropna(subset = ["price"])

# Sort by location and date
df = df.sort_values(["location", "date"])

print(df.head())
print(df.info())

output = BASE_DIR / "data" / "processed" / "clean_prices.csv"
df.to_csv(output, index=False)

print(f"Cleaned data saved to {output}")