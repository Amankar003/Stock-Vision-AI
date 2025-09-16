import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import model

# Streamlit UI
st.title("Indian Stock Predictor")

# Sidebar
symbol = st.sidebar.selectbox("Choose Stock", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"])
start = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))

# Submit Button
if st.sidebar.button("Run Prediction"):
    # Load Data
    df = model.fetch_data(symbol, start)
    df = model.add_features(df)

    # Price Chart
    st.subheader("Stock Price with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_10"], line=dict(color="blue"), name="MA 10"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_50"], line=dict(color="red"), name="MA 50"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Train Model
    st.subheader("Model Training & Evaluation")
    trained_model, acc, features = model.trained_model(df)
    st.write(f" Model trained. Test Accuracy: **{acc:.2f}**")

    # Predict next Day
    pred, prob = model.predict_next_day(df, trained_model, features)
    st.subheader("Next Day Prediction")
    st.metric("Prediction", pred, f"Probability: {prob:.2f}")
else:
    st.info(" Select stock & date, then click **Run Prediction** from sidebar.")
