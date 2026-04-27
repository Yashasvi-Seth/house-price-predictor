import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
file_path = BASE_DIR / "data" / "raw" / "corra.csv"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# finding the line where observations start, which is the line after "OBSERVATION"
start_idx = None
for i, line in enumerate(lines):
    if "OBSERVATION" in line:
        start_idx = i + 1
        break

# Read CSV from the line where observations start
from io import StringIO # StringIO allows us to treat a string as a file-like object,
                        # which is useful for reading the CSV data directly from the string.

# We join the lines from the start index to the end to create a single string that contains the CSV data,
# and then we read it into a DataFrame using pd.read_csv.
csv_txt = "".join(lines[start_idx:])
df = pd.read_csv(StringIO(csv_txt))

# keep only the needed columns and rename them
df = df[["date", "AVG.INTWO"]]
df = df.rename(columns = { "AVG.INTWO": "interest_rate"})

# Convert types
df["date"] = pd.to_datetime(df["date"])
df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors = "coerce")

# Drop rows with missing interest rates
df = df.dropna()

# Convert daily -> monthly data by taking the average interest rate for each month
monthly = (
    df.set_index("date")
    .resample("MS") # resample to monthly frequency
    .mean()        # take the mean of the interest rates for each month
    .reset_index() # reset index to get date back as a column
)

# Save the processed data
output = BASE_DIR / "data" / "processed" / "interest_rates.csv"
monthly.to_csv(output, index=False)

print(monthly.head())
print(f"Saved processed interest rates data to {output}")




