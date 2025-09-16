import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetch stock data

def fetch_data(symbol, start):
    df = yf.download(symbol, start=start)
    df.reset_index(inplace=True)
    return df

# Add Moving Averages

def add_features(df):
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df.dropna(inplace=True)  # remove NaN rows
    return df

# Train Model

def trained_model(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # 1 if next day up, else 0

    features = ["MA_10", "MA_50"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, features

# Predict Next Day

def predict_next_day(df, trained_model, features):
    latest = df[features].iloc[-1:].values
    pred = trained_model.predict(latest)[0]
    prob = trained_model.predict_proba(latest)[0][pred]
    return ("UP" if pred == 1 else "DOWN"), prob
