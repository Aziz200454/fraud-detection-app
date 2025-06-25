import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("🕵️‍♂️ Automated Financial Fraud Detection")

uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.write(df.head())

    # Scale input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict frauds
    predictions = model.predict(X_scaled)
    df["Fraud Prediction"] = predictions

    st.write("✅ Prediction Results:")
    st.dataframe(df)

    st.write(f"🔴 Total Fraudulent Transactions Detected: {sum(predictions)}")
