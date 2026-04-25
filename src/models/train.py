import pandas as pd
from pathlib import Path

# This is used to split the data into training and testing sets
from sklearn.model_selection import train_test_split

# This is the machine learning model we will use
from sklearn.ensemble import RandomForestRegressor

# This is used to evaluate the model's performance
# Measures MAE, which is the average absolute difference between predicted and actual values
from sklearn.metrics import mean_absolute_error
# -------------------------------------------------------------------------------------------------

Base_DIR = Path(__file__).resolve().parents[2]
file_path = Base_DIR / "data" / "processed" / "clean_prices.csv"

df = pd.read_csv(file_path)

# Extract year and month from date, and encode location as categorical variable
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ML models use numerical data, therefore coding the cities as numerical codes
df["location_code"] = df["location"].astype("category").cat.codes

# Define features and target variable
X = df[["year", "month", "location_code"]]
Y = df["price"]

# Split the data into training and testing sets
# here we use 80% of the data for training and 20% for testing
# and set a random state for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create random forest model, fit it to the training data, and make predictions on the test set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)             #Learns relationships: (year, month, location) -> price 
# This is where model studies data.

predictions = model.predict(X_test)

mae = mean_absolute_error(Y_test, predictions)
print(f"Mean Absolute Error: {mae}")