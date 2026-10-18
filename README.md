# 🚗 Car Price Prediction ML

**A machine learning-based web application that predicts the resale market value of used cars based on key vehicle attributes.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

Accurately predicting used car prices requires analyzing various historical vehicle data points. **Car Price Prediction ML** implements a robust machine learning model integrated with a Flask backend to deliver instant, real-time car price estimates through a clean, user-friendly web interface.

### **Key Highlights:**
- **Machine Learning Pipeline:** Trained on automotive dataset attributes to predict vehicle valuations accurately.
- **Flask Web Server:** Provides a seamless interface connecting the ML model inference engine to the web frontend.
- **Interactive UI:** A simple HTML/CSS input form allowing users to select vehicle specs and receive price outputs immediately.

---

## ✨ Features

- 🚘 **Multi-Attribute Input:** Accept features like manufacturing year, total kilometers driven, fuel type, transmission type, and brand.
- ⚡ **Instant Valuation:** Provides fast market price predictions upon form submission.
- 🧹 **Data Preprocessing:** Encodes categorical parameters and scales numeric inputs for reliable model evaluation.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | [Flask](https://flask.palletsprojects.com/) | Handles routing, API endpoints, and serves the HTML templates. |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/), Pandas, NumPy | Model training, data preprocessing, and inference. |
| **Frontend** | HTML5, CSS3 | User-friendly input forms and displaying results. |
| **Model Serialization** | Pickle / Joblib | Saving the trained model to disk and loading it in the Flask app. |

---
## 🧠 How the Model Works
>> Data Cleaning: The dataset is cleaned to handle missing values and remove extreme outliers.

>> Feature Engineering: Categorical variables (e.g., Fuel Type, Transmission) are One-Hot Encoded, and numerical variables (e.g., Kilometers Driven) are scaled.

>> Model Training: A regression model (e.g., Random Forest Regressor / Linear Regression) is trained on the processed data.

>> Deployment: The model is saved as a .pkl file and loaded into app.py to make real-time predictions based on user input.
---
## 👤 Author
Arafat Khan

🌐 Portfolio: arafatkhan.vercel.app

💻 GitHub: @iamthearafatkhan

💼 LinkedIn: iamthearafatkhan

---

## 📄 License
This project is open-source and free for anyone to use, modify, or distribute.
