# 🩺 Diabetes Prediction Web App

A lightweight web application built using **Flask** that predicts whether a person has diabetes based on health metrics. The prediction is powered by a trained **XGBoost classifier** on the Pima Indians Diabetes Dataset, with data balancing using **SMOTE**.

---

## 📌 Project Overview

This app takes in 8 medical inputs from the user (like glucose level, BMI, age, etc.) and uses a machine learning model to classify whether the individual is likely diabetic or not. It’s intended as a proof-of-concept or educational tool — not a diagnostic medical tool.

---

## 🚀 Features

- ✅ Clean UI for user input
- 📊 Machine learning powered prediction (XGBoost)
- 🔄 Handles imbalanced datasets using SMOTE
- 🧠 Trained on the well-known Pima Indians Diabetes Dataset
- ⚡ Fast, responsive, and easy to deploy

---

## 🧪 Model Details

| Property        | Value                                 |
|----------------|----------------------------------------|
| Algorithm       | XGBoost Classifier                    |
| Preprocessing   | SMOTE for class balancing             |
| Accuracy        | ~75%                                  |
| Output          | Prediction (Diabetic / Not)    |

---


## Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate   # On Windows
or
source venv/bin/activate   # On macOS/Linux

---

## Install dependencies

pip install -r requirements.txt

Run the Flask app

python app.py
Visit http://127.0.0.1:5000 to use the app in your browser.