import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import joblib
import pickle
import shap
import openai
import base64

MODEL_FILE = "telco-model-xgb.joblib"
LE_FILE_Y = "le-preprocessor-y.joblib"
MCLE_FILE_PKL = "mcle-preprocessor-x.pkl"
SHAP_FILE_PKL = "shap-explainer.pkl"

CHART_FILE_NAME = "display_chart.png"

features = {
    "tenure": 12,
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "OnlineSecurity": ["Yes", "No"],
    "TechSupport": ["Yes", "No"],
    "OnlineBackup": ["Yes", "No"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "StreamingMovies": ["Yes", "No"],
    "StreamingTV": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "DeviceProtection": ["Yes", "No"],
    "gender": ["Male", "Female"],
    "SeniorCitizen": ["No", "Yes"],
    "Dependents": ["Yes", "No"],
    "Partner": ["Yes", "No"],
}

mcle = pickle.load(open(MCLE_FILE_PKL, "rb"))
model = joblib.load(MODEL_FILE)
le_y = joblib.load(LE_FILE_Y)
explainer = pickle.load(open(SHAP_FILE_PKL, "rb"))

st.title("Customer Churn Prediction")

client = openai.OpenAI(
    api_key="94787749-a159-41b3-8e9d-7eda5f7e7d66",
    base_url="https://api.sambanova.ai/v1",
)

with st.form("churn_form"):
    st.subheader("Enter Customer Details")
    col8, col9 = st.columns([2, 1])
    col1, col2, col3 = st.columns(3)
    col4, col5, col6, col7 = st.columns(4)
    user_inputs = {}

    with col8:
        USER_NAME = st.text_input("Customer Name", value="Viknesh")
    with col9:
        user_inputs['OnlineSecurity'] = st.selectbox("Online Security", features['OnlineSecurity'])

    with col1:
        user_inputs['tenure'] = st.number_input("Tenure in months", min_value=1, step=1, value=12)
        user_inputs['Contract'] = st.selectbox("Contract", features['Contract'])
    with col2:
        user_inputs['MonthlyCharges'] = st.number_input("Monthly Charges", value=50.0)
        user_inputs['PaymentMethod'] = st.selectbox("Payment Method", features['PaymentMethod'])
    with col3:
        user_inputs['TotalCharges'] = st.number_input("Total Charges", value=600.0)
        user_inputs['InternetService'] = st.selectbox("Internet Service", features['InternetService'])

    with col4:
        user_inputs['TechSupport'] = st.selectbox("Tech Support", features['TechSupport'])
        user_inputs['StreamingMovies'] = st.selectbox("Streaming Movies", features['StreamingMovies'])
        user_inputs['gender'] = st.selectbox("Gender", features['gender'])

    with col5:
        user_inputs['OnlineBackup'] = st.selectbox("Online Backup", features['OnlineBackup'])
        user_inputs['StreamingTV'] = st.selectbox("Streaming TV", features['StreamingTV'])
        user_inputs['SeniorCitizen'] = st.selectbox("Senior Citizen", features['SeniorCitizen'])

    with col6:
        user_inputs['PaperlessBilling'] = st.selectbox("Paperless Billing", features['PaperlessBilling'])
        user_inputs['PhoneService'] = st.selectbox("PhoneService", features['PhoneService'])
        user_inputs['Dependents'] = st.selectbox("Dependents", features['Dependents'])

    with col7:
        user_inputs['MultipleLines'] = st.selectbox("Multiple Lines", features['MultipleLines'])
        user_inputs['DeviceProtection'] = st.selectbox("Device Protection", features['DeviceProtection'])
        user_inputs['Partner'] = st.selectbox("Partner", features['Partner'])

    submitted = st.form_submit_button("Predict and Explain")

if submitted:
    user_inputs = {k: [v] for k, v in user_inputs.items()}

    test = pd.DataFrame(user_inputs)
    pre_test = pd.DataFrame(user_inputs)
    test = mcle.transform(test)
    test['SeniorCitizen'] = test['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    display_data = pd.concat([pre_test, test])
    display_data.index = ['Before', 'After']
    display_data = display_data[model.get_booster().feature_names]
    test = test[model.get_booster().feature_names]
    st.write("Before/After Preprocessing the data:", display_data)
    st.write("Will the customer churn?: ", le_y.inverse_transform(model.predict(test))[0])

    shap_values = explainer(test)
    ax = shap.waterfall_plot(shap_values[0], show=False, max_display=30)
    ax.set_title(f'SHAP result for the customer: {USER_NAME}')
    aspect_ratio_multiplier = 1.25
    ax.figure.set_size_inches(16 * aspect_ratio_multiplier, 9 * aspect_ratio_multiplier)
    plt.savefig(CHART_FILE_NAME)

    st.image(CHART_FILE_NAME)

    response = None

    try:
        img = None

        with open(CHART_FILE_NAME, "rb") as img:
            img = base64.b64encode(img.read()).decode()

        PROMPT = f"give me only a small conclusion with the provided image about the major contributing features specific for the customer in the prediction, and explain the content related with the customer churning. and make the conclusions into points wise instead of a paragraph. instead of the title conclusion, name it SHAP Inference. also the customer name if {USER_NAME}. Do Not make a point about what image is. give explanation to each points you make. dont just give blank points. group similar kind of values and give a grouped points and reason."
        response = client.chat.completions.create(
            model='Llama-3.2-90B-Vision-Instruct',
            messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                   {"type": "image_url",
                                                    "image_url": {"url": f"data:image/png;base64,{img}"}}]}],
            temperature=0.1,
            top_p=0.1
        )
        st.markdown(response.choices[0].message.content, unsafe_allow_html=True)
    except openai.RateLimitError as err:
        st.write("API has exceeded the request limit. Please wait for 1 minute before trying again.")

# %%
