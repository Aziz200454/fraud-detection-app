 import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Fraud Detection", layout="centered", page_icon="🕵️")

# Load trained model and expected feature names
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("features.pkl")

# App Title
st.markdown("<h1 style='text-align: center; color: navy;'>🕵️ Automated Financial Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ✅ Check if uploaded file has correct columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            st.error(f"🚫 Uploaded file is missing required columns: {missing_cols}")
            st.stop()

        # ✅ Reorder columns to match model training
        df = df[feature_names]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Make predictions
        predictions = model.predict(X_scaled)
        df["Fraud Prediction"] = predictions

        # Show results
        st.success("✅ Predictions generated successfully!")
        st.subheader("🔍 Prediction Results")
        st.dataframe(df, use_container_width=True)

        # Show summary
        total = len(df)
        frauds = int(sum(predictions))
        non_frauds = total - frauds

        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Fraudulent", frauds)
        col3.metric("Legitimate", non_frauds)

        # Bar chart
        st.subheader("📉 Fraud vs Legitimate Transactions")
        st.bar_chart(pd.Series([non_frauds, frauds], index=["Legitimate", "Fraudulent"]))
