# 📊 Customer Churn Prediction

This project predicts whether a telecom customer is likely to **churn (leave the service)** or **stay** using machine learning.  
It also includes a **Streamlit web app** that makes the model interactive and user-friendly.

---

## 🚀 Features
- **Data Preprocessing & Encoding** → Cleaned and prepared customer dataset.  
- **Model Training** → Built and trained machine learning models for churn prediction.  
- **Model Persistence** → Saved the trained model (`Churn_Model.pkl`) and encoders (`encoders.pkl`) for reuse.  
- **Streamlit App** → Interactive UI (`app.py`) where users enter customer details and get real-time predictions.  

---

## 🛠️ Tech Stack
- **Python** → Pandas, NumPy, Scikit-learn  
- **Streamlit** → For interactive web app  
- **Pickle** → For saving model and encoders  

---

## 📂 Project Structure
├── app.py # Streamlit web app.  
├── Churn_Model.pkl # Trained ML model.   
├── encoders.pkl # Encoders for categorical features.  
├── requirements.txt # Python dependencies.  
└── README.md # Project documentation.
