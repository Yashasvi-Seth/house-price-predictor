import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

#------------------------------------------------
# Path
#------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "processed" / "clean_prices.csv"
rates_path = BASE_DIR / "data" / "processed" / "interest_rates.csv"
pop_path = BASE_DIR / "data" / "raw" / "population.csv"
model_path = BASE_DIR / "src" / "models" / "model.pkl"
#------------------------------------------------
# Load data and model
#------------------------------------------------
df = pd.read_csv(data_path)
rates = pd.read_csv(rates_path)
pop = pd.read_csv(pop_path)

df["date"] = pd.to_datetime(df["date"])
rates["date"] = pd.to_datetime(rates["date"])

df["year"] = df["date"].dt.year
df = df.merge(rates, on="date", how="left")
df = df.merge(pop, on=["location", "year"], how="left")

# ------------------------------------------------
# Feature Engineering (match training pipeline)
# ------------------------------------------------
df = df.sort_values(["location", "date"])

group = df.groupby("location")

# Population growth
df["pop_growth"] = group["population"].pct_change()

# Price lag features
df["price_lag_1"] = group["price"].shift(1)

df["price_3_mean"] = (
    group["price"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["price_lag_12"] = group["price"].shift(12)

# Growth features
df["price_growth_1"] = (
    (df["price"] - df["price_lag_1"]) / df["price_lag_1"]
)

df["price_growth_12"] = (
    (df["price"] - df["price_lag_12"]) / df["price_lag_12"]
)

# Fill first rows that naturally become NaN
df = df.bfill()

model = joblib.load(model_path)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Encode location
df["location_code"] = df["location"].astype("category").cat.codes

location_map = dict(enumerate(df["location"].astype("category").cat.categories))
reverse_map = {v: k for k, v in location_map.items()}

#------------------------------------------------
# Title and description
#------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown("""
<style>
div[data-baseweb="select"] > div {
    cursor: pointer !important;
}

div[data-baseweb="select"] * {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

st.title("House Price Predictor")
st.caption("AI-powered forecasting using historical trends, rates, and population data.")

#------------------------------------------------
# Sidebar for user input
#------------------------------------------------
st.sidebar.header("Input Features")
location = st.sidebar.selectbox("Location", options=sorted(df["location"].unique()))

latest_year = int(df["year"].max())

st.write("Uses latest known market conditions as baseline.")
year = st.sidebar.slider(
    "Scenario Year",
    min_value=latest_year - 5,
    max_value=latest_year + 5,
    value=latest_year + 1
)

st.sidebar.caption("Choose the year you want to estimate house prices for.")

month = st.sidebar.slider("Month", min_value=1, max_value=12, value=1)

city_df = df[df["location"] == location].sort_values("date")

latest = city_df.iloc[-1] # Get the latest available data for the selected city,
                        # which will be used to fill in the features for prediction.

input_data = pd.DataFrame({
    "year": [year],
    "month": [month],
    "location_code": [reverse_map[location]],
    "interest_rate": [latest["interest_rate"]],
    "population": [latest["population"]],
    "pop_growth": [latest["pop_growth"]],
    "price_lag_1": [latest["price"]],
    "price_3_mean": [city_df["price"].tail(3).mean()],
    "price_lag_12": [city_df["price"].iloc[-12]] if len(city_df) >= 12 else [latest["price"]],
    "price_growth_1": [0],  # Placeholder, not used for prediction
    "price_growth_12": [0]  # Placeholder, not used for prediction
    
})

#------------------------------------------------
# Make prediction
#------------------------------------------------
predicted_price = model.predict(input_data)[0]
st.metric("Predicted Price", f"${predicted_price:,.0f}")

# ------------------------------------------------
# Historical Chart
# ------------------------------------------------
st.subheader("Historical Prices")

fig = px.line(
    city_df,
    x="date",
    y="price",
    title=f"{location} Price Trend"
)

st.plotly_chart(fig, use_container_width=True)

#------------------------------------------------
# Raw data
#------------------------------------------------
with st.expander("Show Raw Data"):
    st.dataframe(city_df.tail(20))
