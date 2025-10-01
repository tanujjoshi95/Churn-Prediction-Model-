# 📊 Customer Churn Prediction

This project predicts whether a telecom customer is likely to **churn (leave the service)** or **stay** using machine learning.  
It also includes a **Streamlit web app** that makes the model interactive and user-friendly.

---

## 🚀 Features
- **Exploratory Data Analysis (EDA):** Performed in Jupyter Notebook to understand churn patterns.  
- **Feature Engineering & Encoding:** Prepared categorical and numerical features.  
- **Model Training:** Built and evaluated ML models in the notebook.  
- **Model Persistence:** Saved the trained model (`Churn_Model.pkl`) and encoders (`encoders.pkl`) for reuse.  
- **Streamlit App:** Interactive UI (`app.py`) where users enter customer details and get real-time predictions.  

---

## 🛠️ Tech Stack
- **Python** → Pandas, NumPy, Scikit-learn  
- **Streamlit** → For interactive web app  
- **Pickle** → For saving model and encoders  
- **Jupyter Notebook** → For analysis and model development  

---

## 📂 Project Structure
├── app.py # Streamlit web app.  
├── churn_prediction.ipynb # Jupyter Notebook with EDA and model training.  
├── Churn_Model.pkl # Trained ML model.   
├── encoders.pkl # Encoders for categorical features.   
└── README.md # Project documentation.
