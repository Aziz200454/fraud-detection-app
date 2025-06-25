import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Fraud Detection", layout="centered", page_icon="🕵️")

# --- HEADER ---
st.markdown("""
    <div style='background-color:#1f77b4; padding:20px; border-radius:10px'>
        <h2 style='color:white; text-align:center;'>🕵️ Financial Fraud Detection Dashboard</h2>
    </div>
    <br>
""", unsafe_allow_html=True)

# --- SIDEBAR INSTRUCTIONS ---
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
**Upload your transaction data (.csv):**
- Must match trained features
- Predictions will label each row as **Fraudulent** or **Legitimate**

**Outputs include:**
- Fraud summary statistics
- Interactive bar chart
- Downloadable results
""")

# Load model and features
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("features.pkl")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Your CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Check and reorder columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            st.error(f"🚫 Missing required columns: {missing_cols}")
            st.stop()
        df = df[feature_names]

        # Scale and predict
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        predictions = model.predict(X_scaled)
        df["Fraud Prediction"] = predictions

        # Show results
        st.success("✅ Predictions generated successfully!")
        st.subheader("🔍 Prediction Results")
        st.dataframe(df, use_container_width=True)

        # Summary stats
        total = len(df)
        frauds = int(sum(predictions))
        non_frauds = total - frauds

        st.markdown("### 📈 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Frauds", frauds)
        col3.metric("Legit", non_frauds)

        # Bar chart
        st.subheader("📉 Fraud vs Legitimate Transactions")
        st.bar_chart(pd.Series([non_frauds, frauds], index=["Legitimate", "Fraudulent"]))

        # Download results
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("📄 Please upload a CSV file to begin.")

# --- FOOTER ---
st.markdown("<hr><div style='text-align:center; color: gray;'>© 2025 Fraud Detection System | Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
