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

# -----------------------------------------------

# Adding lag(recent prices from the last few months) features: 
group = df.groupby("location")

df["price_lag_1"] = group["price"].shift(1)  # Price from the previous month
df["price_3_mean"] = group["price"].shift(1).rolling(window = 3).mean()
df["price_lag_12"] = group["price"].shift(12)  # Price from the same month last year

# Growth rate features:
df["price_growth_1"] = (df["price"] - df["price_lag_1"]) / df["price_lag_1"]
df["price_growth_12"] = (df["price"] - df["price_lag_12"]) / df["price_lag_12"]

# -----------------------------------------------

# Define features and target variable
X = df[[
    "year",
    "month",
    "location_code",
    "price_lag_1",
    "price_3_mean",
    "price_lag_12",
    "price_growth_1",
    "price_growth_12"]]

Y = df["price"]

# Split the data into training and testing sets
# here we use 80% of the data for training and 20% for testing
# and set a random state for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create random forest model, fit it to the training data, and make predictions on the test set
model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1)  # n_jobs=-1 uses all available CPU cores for faster training

model.fit(X_train, Y_train)             #Learns relationships: (year, month, location) -> price 
# This is where model studies data.

predictions = model.predict(X_test)

mae = mean_absolute_error(Y_test, predictions)
print(f"Mean Absolute Error: {mae}")


# -------------------------------------------------------------------------------------------------
# This version not just uses the date and location, but also incorporates recent price trends
# (lag features) and growth rates, which can help the model capture more complex
# patterns in the data. Bringing the MAE down by approximately 65-67% compared to the previous
# version. This shows that the additional features provide valuable information
# for predicting house prices.