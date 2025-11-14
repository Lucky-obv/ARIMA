import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

st.title("📈 Stock Price Forecasting Dashboard")
st.write("Enter a stock ticker (e.g., 1299.HK, 0005.HK, AAPL, TSLA)")

# ----------------------------
# USER INPUT
# ----------------------------
ticker = st.text_input("Enter Stock Symbol:", value="1299.HK")

start_date = st.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.date_input("End Date", dt.date.today())

# ----------------------------
# FUNCTION ADF TEST
# ----------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    if result[1] < 0.05:
        return "✅ The series is **stationary**"
    else:
        return "❌ The series is **not stationary**"

# ----------------------------
# MAIN LOGIC
# ----------------------------
if st.button("Generate Forecast"):

    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ No data found. Check ticker symbol.")
    else:
        # Clean dataframe
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()

        st.subheader("Raw Data")
        st.dataframe(df)

        # Stationarity check
        st.subheader("ADF Stationarity Test")
        st.write(check_stationarity(df["Close"]))

        # ARIMA model
        st.subheader("ARIMA Forecasting")

        try:
            model = ARIMA(df["Close"], order=(5, 0, 0))
            model_fit = model.fit()

            forecast_steps = 10
            forecast = model_fit.forecast(steps=forecast_steps)

            # Dates for forecast
            dates = pd.date_range(
                start=df["Date"].iloc[-1],
                periods=forecast_steps + 1,
                freq="B"
            )[1:]

            # Plot
            fig = plt.figure(figsize=(10, 5))
            plt.plot(df["Date"], df["Close"], label="Actual Prices")
            plt.plot(dates, forecast, label="Predicted Prices",
                     linestyle='dashed', color='red')
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.title(f"Actual vs Predicted Prices ({ticker})")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Model error: {e}")

