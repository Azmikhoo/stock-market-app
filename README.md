
# Stock Price Prediction App

A web application built with Streamlit to predict stock prices using machine learning models.  
This project uses historical stock data and several time-series features to forecast the next day's closing price.

## Dataset

The dataset is obtained using Yahoo Finance API (yfinance).  
It contains historical stock data including:

- Open
- High
- Low
- Close
- Volume

Example stock used in this project: AAPL (Apple Inc).

## Feature Engineering

Several time-series features are created from the closing price:

- Lag_1
- Lag_2
- Lag_3
- Lag_5
- Rolling_Mean_5
- Rolling_Mean_10

The target variable is the next-day closing price.

## Models

Three machine learning models are used:

- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter tuning is performed using GridSearchCV with TimeSeriesSplit.

## Evaluation Metrics

The models are evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## Deployment

The application is deployed using Streamlit.

Users can input a stock ticker and view predicted stock prices based on the trained model.

## How to Run

Clone the repository:

git clone https://github.com/Azmikhoo/stock-streamlit-app.git

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py
