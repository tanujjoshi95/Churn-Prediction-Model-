import streamlit as st
import pickle
import pandas as pd

# Load the saved model
with open("Churn_Model.pkl", "rb") as fs:
    model = pickle.load(fs)

loaded_model = model["model"]
feature_names = model["features"]

# Load encoders for categorical columns
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("📊Customer Churn Prediction App")
st.write("Enter customer details to predict if they are likely to churn.")

with st.form("churn_form"):

    # Group 1: Demographics

    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen (1 = Yes, 0 = No)", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)

    # Group 2: Services
    
    st.subheader("📡 Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # Group 3: Billing

    st.subheader("💳 Billing & Payment")
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
   
    TotalCharges=MonthlyCharges*tenure
    # st.text(f"Total Charges : {TotalCharges}")
    

    # Submit button
    submitted = st.form_submit_button(" Predict")


if submitted:
    input_data_d = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService':InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}


    input_data= pd.DataFrame([input_data_d])

    # Encode categorical features using saved encoders
    for column, encoder in encoders.items():
        input_data[column] = encoder.transform(input_data[column])

    # Make prediction
    prediction = loaded_model.predict(input_data)[0]
    pred_prob = loaded_model.predict_proba(input_data)[0][1]  # probability of churn

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn(Leave).")
    else:
        st.success(f"✅ This customer is not likely to churn(Leave).")
    
with st.sidebar:
    st.markdown("### 🧠 About This Project")
    st.write(
        """
This web app demonstrates a **Customer Churn Prediction Model** built as part of my data science learning journey.

The goal is to identify which customers are at **risk of leaving**, using historical customer behavior and demographic data.
        """
    )

    st.markdown("#### 🎯 What This App Can Do")
    st.write(
        """
- Takes customer details as input  
- Applies a trained ML model  
- Predicts whether the customer will **Churn** or **Stay**  
- Shows prediction probability  
- Lets you explore patterns and insights  
        """
    )

    st.markdown("#### ⚙️ Technologies Used")
    st.write(
        """
- **Python**, **NumPy**, **Pandas**  
- **Matplotlib**, **Seaborn** (visualizations)  
- **scikit-learn** (ML models)  
- **Imbalanced-Learn (SMOTE)**  
- **XGBoost**, **LightGBM**, **Random Forest**, **Decision Tree** 
- **Pickle** for model loading
- **Streamlit** for interactive UI  
        """
    )

    st.markdown("#### 🔬 ML Workflow")
    st.write(
        """
1. **Data Preprocessing**
   - Label encoding  
   - Handling missing values  
   - Scaling & feature engineering  

2. **Handling Imbalance**
   - Oversampling minority class using **SMOTE**  

3. **Model Training**
   - Random Forest  
   - XGBoost  
   - LightGBM  
   - Decision Tree  
   - Hyperparameter tuning (RandomizedSearchCV)  

4. **Model Evaluation**
   - Accuracy & precision  
   - Classification report  
   - Confusion matrix  
        """
    )

    st.markdown("---")
    st.caption("Made by Tara · Data Science & Android dev learner")