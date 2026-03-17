Autism Screening ML Web App

A full-stack machine learning web application that predicts the likelihood of autism traits using behavioral and demographic inputs.

This project demonstrates an end-to-end ML pipeline, from data preprocessing and model training to deployment via a FastAPI backend and Streamlit frontend.

Features
- AQ-10 inspired behavioral screening questionnaire
- Real-time prediction using a trained ML model
- Probability-based risk scoring
- Streamlit-based interactive UI
- FastAPI backend for model inference
- End-to-end pipeline (data → model → API → frontend)

Tech Stack
Frontend: Streamlit
Backend: FastAPI
ML: scikit-learn (Logistic Regression, Random Forest)
Data Processing: pandas, sklearn preprocessing
Model Serving: joblib

How It Works
1. User inputs behavioral responses
2. Inputs are encoded into numerical features
3. Backend loads trained model + scaler
4. Model outputs probability score
5. Frontend displays risk interpretation

Model Details
Dataset: Autism Screening Dataset (Kaggle)
Features:
AQ-style behavioral scores (A1–A10)
- Age
- Gender
- Medical history indicators

Models tested:
- Logistic Regression
- Random Forest (selected for performance)

Disclaimer

This tool is for screening purposes only and is not a medical diagnosis. Always consult a qualified professional.

Motivation

This project was inspired by personal experience navigating autism care systems and aims to make early guidance more accessible for families.
