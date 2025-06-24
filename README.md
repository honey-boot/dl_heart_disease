# 💓 Heart Disease Prediction App

This is a simple Deep Learning project built with TensorFlow and Streamlit to predict the presence of heart disease using clinical parameters.

## ✅ Features

- Accepts user inputs for 13 medical parameters
- Predicts heart disease using a trained deep learning model
- Built using TensorFlow and Streamlit

## 📁 Files

- `app.py`: Streamlit web app
- `heart_disease_model.h5`: Saved model file
- `scaler.pkl`: StandardScaler (used only if model was trained on scaled data)
- `requirements.txt`: Required Python packages

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
