import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Fraud Detection", layout="centered", page_icon="🕵️")

# Load trained model and feature list
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("features.pkl")

# App Title
st.markdown("<h1 style='text-align: center; color: navy;'>🕵️ Automated Financial Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load user data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Check for matching features
    if not all(col in df.columns for col in feature_names):
        st.error("Uploaded file does not contain the expected columns used during training. Please upload a valid dataset.")
        st.stop()

    # Reorder columns to match training data
    df = df[feature_names]

    # Scale input
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict
    predictions = model.predict(X_scaled)
    df["Fraud Prediction"] = predictions

    # Results
    st.success("✅ Predictions generated successfully!")
    st.subheader("🔍 Prediction Results")
    st.dataframe(df, use_container_width=True)

    # Summary
    total = len(df)
    frauds = sum(predictions)
    non_frauds = total - frauds

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Fraudulent", frauds)
    col3.metric("Legitimate", non_frauds)

    st.subheader("📉 Fraud vs Legitimate Chart")
    st.bar_chart(pd.Series([non_frauds, frauds], index=["Legitimate", "Fraudulent"]))

    st.download_button(
        label="⬇️ Download Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("📄 Please upload a CSV file with the correct features to get started.")

