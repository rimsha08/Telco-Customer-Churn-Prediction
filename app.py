import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("telco_churn_model.pkl")

# App title and info
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

with st.sidebar:
    st.info("ℹ️ **About This App**")
    st.markdown("""
    This app uses a **Logistic Regression** model trained on the **Telco Customer Churn** dataset.

    **Features include:**
    - Demographics
    - Account info
    - Services usage

    The model is built using **Scikit-learn Pipelines** and supports end-to-end preprocessing.
    """)

st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict whether a customer will **churn** based on their account and service information.")


# Function to collect user input
def user_input_features():
    with st.form("customer_form"):
        st.subheader("📝 Enter Customer Information")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col2:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)

        submit = st.form_submit_button("📈 Predict Churn")

        if submit:
            return pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }])
        else:
            return None


# Collect user input
input_df = user_input_features()

# Predict if input is available
if input_df is not None:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1 or prediction == "Yes":
        st.error(f"⚠️ Customer is likely to **CHURN**.\n\nConfidence: {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to **STAY**.\n\nConfidence: {1 - probability:.2%}")


