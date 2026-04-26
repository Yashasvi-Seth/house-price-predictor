# Ontario House Price Predictor

A machine learning project focused on predicting house prices across Ontario cities using historical housing data and economic, demographic, and regional indicators.

This project is designed as a real-world end-to-end data science pipeline, combining:

* Data ingestion
* Data cleaning & preprocessing
* Feature engineering
* Predictive modeling
* Model evaluation
* Interactive dashboard visualization

---

# Project Goal

To estimate housing prices for Ontario locations based on:

* Historical benchmark prices
* City / region
* Time (year / month)
* Population growth
* Immigration trends
* Tax rates
* Interest rates
* Logistics / infrastructure access
* Employment & income indicators
* Other macroeconomic signals

---

# Current Features

## Implemented

* Python virtual environment setup
* Structured project architecture
* Housing dataset ingestion (CSV)
* Data cleaning pipeline
* Benchmark price extraction
* Date feature engineering
* Baseline regression model using Random Forest Regressor
* Model evaluation using MAE (Mean Absolute Error)

---

# Upcoming Features

## Data Expansion

* Population by city
* Population growth rate
* Immigration statistics
* Mortgage / Bank of Canada interest rates
* Property tax rates
* Average income by region
* Unemployment rate
* Infrastructure / transit access
* Accident / safety statistics
* Supply inventory (new listings)

---

## Machine Learning Improvements

* Lag features (previous month prices)
* Rolling averages
* Seasonal indicators
* XGBoost Regressor
* CatBoost Regressor
* Hyperparameter tuning
* Cross-validation
* Ensemble models
* Future price forecasting

---

## Dashboard Features

* Interactive Ontario map
* City-wise price trends
* Price prediction tool
* Feature importance visualization
* Market comparison by city
* Historical trend charts
* Scenario simulation (tax increase / rate cuts)

---

## Engineering Improvements

* Automated dataset updates
* Scheduled monthly retraining
* Model versioning
* Logging system
* Config files
* Docker support
* Cloud deployment

---

# Project Structure

house-price-predictor/
│── data/
│   ├── raw/
│   └── processed/
│
│── models/
│
│── notebooks/
│
│── src/
│   ├── ingestion/
│   ├── processing/
│   ├── features/
│   └── models/
│
│── dashboard/
│
│── requirements.txt
│── README.md

---

# Tech Stack

## Core

* Python 3.11
* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* XGBoost *(planned)*
* CatBoost *(planned)*

## Visualization

* Matplotlib
* Seaborn
* Plotly

## Dashboard

* Streamlit

## Utilities

* Pathlib
* Joblib
* Requests

# 💻 Requirements

## Recommended Python Version

Python 3.11.x
(Stable for ML libraries)

## Install Dependencies

pip install -r requirements.txt

---

# ▶️ Running the Project

## 1. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 2. Run Data Pipeline

```bash
python src/processing/load_data.py
```

---

## 3. Train Model

```bash
python src/models/train.py
```

---

## 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📊 Current Model

### Baseline Model

Random Forest Regressor using:

* Year
* Month
* Location Code

### Target Variable

* Benchmark Housing Price

### Evaluation Metric

* MAE (Mean Absolute Error)

---

# 🧠 Why This Project Matters

Ontario housing is influenced by multiple factors beyond raw historical prices.

This project aims to combine:

* Time-series behavior
* Regional demand
* Economic conditions
* Demographic pressure

into a practical predictive system.

---

# Development Roadmap

## Phase 1

* Baseline model - DONE
* Clean pipeline - DONE

## Phase 2

* Add macroeconomic features
* Improve model accuracy

## Phase 3

* Dashboard launch

## Phase 4

* Automated retraining & deployment

---

# Software Design Principles Used

* Modular folder structure
* Separation of concerns
* Reproducible environments
* Configurable pipeline
* Scalable architecture
* Clean data flow

---

# Author

Yashasvi Seth 
---
