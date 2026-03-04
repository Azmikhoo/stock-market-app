import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Ridge Stock Forecast", layout="centered")

st.title("Next Day Stock Forecast Using Ridge Regression")

# Load artifacts
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

# =========================
# USER INPUT SECTION
# =========================

ticker = st.text_input("Enter Stock Ticker", "AAPL")

today = datetime.date.today()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))

with col2:
    end_date = st.date_input("End Date", today)

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict"):

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this ticker and date range.")
    elif len(df) < 50:
        st.error("Please select a longer date range, minimum 50 days.")
    else:
        # =========================
        # FEATURE ENGINEERING
        # =========================

        df["Lag_1"] = df["Close"].shift(1)
        df["Lag_2"] = df["Close"].shift(2)
        df["Lag_3"] = df["Close"].shift(3)
        df["Lag_5"] = df["Close"].shift(5)
        df["Rolling_Mean_5"] = df["Close"].rolling(5).mean()
        df["Rolling_Mean_10"] = df["Close"].rolling(10).mean()

        df = df.dropna()

        if len(df) == 0:
            st.error("Not enough data after feature engineering.")
        else:
            latest = df.iloc[-1]

            try:
                X_latest = latest[feature_columns].values.reshape(1, -1)
            except KeyError:
                st.error("Feature mismatch between training and deployment.")
                st.write("Model expects:", feature_columns)
                st.write("Available columns:", df.columns.tolist())
                st.stop()

            X_scaled = scaler.transform(X_latest)

            prediction = float(model.predict(X_scaled)[0])
            last_close = float(latest["Close"])

            direction = "UP" if prediction > last_close else "DOWN"

            # =========================
            # DISPLAY RESULT
            # =========================

            st.subheader("Prediction Result")
            st.write("Last Close Price:", round(last_close, 2))
            st.write("Predicted Next Close:", round(prediction, 2))
            st.write("Predicted Direction:", direction)

            # =========================
            # VISUALIZATION
            # =========================

            st.subheader("Last 60 Days Closing Price")
            st.line_chart(df["Close"].tail(60))