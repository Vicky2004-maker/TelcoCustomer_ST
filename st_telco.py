import os
import pickle
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("---")

STACKING_MODEL_PATH = "stacking.joblib"
PREPROCESSOR_PATH = "preprocessing_transformer.pkl"
SHAP_MODEL_PATH = "shap-explainer.joblib"

if os.path.exists(STACKING_MODEL_PATH):
    stacking_model = joblib.load(STACKING_MODEL_PATH)
    st.success("Loaded stacking model successfully.")
else:
    st.error(f"Stacking model not found at {STACKING_MODEL_PATH}.")
    stacking_model = None

if os.path.exists(PREPROCESSOR_PATH):
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    st.success("Loaded preprocessing transformer successfully.")
else:
    st.error(f"Preprocessing transformer not found at {PREPROCESSOR_PATH}.")
    preprocessor = None

mode = st.radio("Select Input Mode", ["Manual Input", "Upload File"])

expected_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

if mode == "Manual Input":
    st.markdown("## Enter Customer Details")
    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    senior_citizen = col2.selectbox("Senior Citizen", ["Yes", "No"])
    senior_citizen_val = 1 if senior_citizen == "Yes" else 0
    partner = col3.selectbox("Partner", ["Yes", "No"])
    dependents = col4.selectbox("Dependents", ["Yes", "No"])

    col5, col6, col7 = st.columns(3)
    tenure = col5.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    phone_service = col6.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = col7.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    col8, col9, col10 = st.columns(3)
    internet_service = col8.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = col9.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = col10.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    col11, col12, col13 = st.columns(3)
    device_protection = col11.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = col12.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = col13.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    col14, col15, col16 = st.columns(3)
    streaming_movies = col14.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = col15.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = col16.selectbox("Paperless Billing", ["Yes", "No"])

    col17, col18, col19 = st.columns(3)
    payment_method = col17.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = col18.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = col19.number_input("Total Charges", min_value=0.0, value=100.0)

    user_input = {
        "gender": gender,
        "SeniorCitizen": senior_citizen_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
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
    }
    st.markdown("---")
    st.write("### Input Summary")
    input_df = pd.DataFrame([user_input])
    st.dataframe(input_df)

else:
    st.markdown("## Upload a CSV or Excel File")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("### Uploaded Data")
            st.dataframe(input_df)
            num_rows = len(input_df)
            st.write(f"File has {num_rows} rows.")
            row_start = st.number_input("Start Row (0-indexed)", min_value=0, max_value=num_rows-1, value=0, step=1)
            row_end = st.number_input("End Row (0-indexed)", min_value=row_start, max_value=num_rows-1, value=num_rows-1, step=1)
            input_df = input_df.iloc[int(row_start): int(row_end)+1]
            st.write(f"### Selected rows: {row_start} to {row_end}")
            st.write("### Selected Data")
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            input_df = None
    else:
        input_df = None

if st.button("Predict Churn") and input_df is not None:
    if (stacking_model is None) or (preprocessor is None):
        st.error("Model or preprocessor not loaded correctly.")
    else:
        for col in expected_features:
            if col not in input_df.columns:
                if col == "Churn":
                    input_df[col] = "Yes"
                else:
                    st.error(f"Missing column: {col}")
                    st.stop()
        input_df = input_df[expected_features]
        preprocessed_data = preprocessor.transform(input_df)
        try:
            preprocessed_df = pd.DataFrame(preprocessed_data, columns=input_df.columns)
            preprocessed_df = preprocessed_df.drop("Churn", axis=1)
            preprocessed_data = preprocessed_df.values
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()
        prediction = stacking_model.predict(preprocessed_data)
        prediction_proba = stacking_model.predict_proba(preprocessed_data)[:, 1]
        results_df = input_df.copy().drop("Churn", axis=1)
        results_df["Prediction"] = ["Churn" if p == 1 else "No Churn" for p in prediction]
        results_df["Churn Probability"] = prediction_proba
        st.markdown("## Prediction Results")
        st.dataframe(results_df)

        if os.path.exists(SHAP_MODEL_PATH):
            with st.spinner("Loading SHAP explainer and computing SHAP values..."):
                explainer = joblib.load(SHAP_MODEL_PATH)
                shap_values = explainer(preprocessed_data)
            st.markdown("### Beeswarm Plot")
            plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

            st.markdown("### Dependence Plot")
            selected_feature = st.selectbox("Select Feature for Dependence Plot", preprocessed_df.columns, key="dep_feature")
            with st.spinner("Computing dependence plot..."):
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(selected_feature, shap_values.values, features=preprocessed_df, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            st.markdown("### Force Plot (for first instance)")
            with st.spinner("Generating force plot..."):
                try:
                    force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0], preprocessed_df.iloc[0], matplotlib=True)
                    plt.figure(figsize=(8, 3))
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.info(f"Force plot not available: {e}")
        else:
            st.info("SHAP explainer not found. Skipping explanation.")
