# Heart_Disease_Predictor_ML_app
# Medical Insurance Charges Prediction

This project is a Machine Learning–based Medical Insurance Charges Prediction System that uses patient information to estimate insurance costs. The model is trained using Random Forest Regression and deployed as an interactive Streamlit web application.

---

## Project Overview

Medical insurance charges depend on multiple factors such as age, smoking habits, BMI, and lifestyle. This project analyzes these factors, performs feature engineering and statistical tests, and builds a predictive model to estimate medical charges accurately.

The final solution is deployed as a Streamlit web app, allowing users to input patient details and instantly get predicted insurance charges.

---

## Features Used

The model uses the following features:

* age – Age of the person
* is_female – Gender (1 = Female, 0 = Male)
* bmi – Body Mass Index
* children – Number of children
* is_smoker – Smoking status (1 = Smoker, 0 = Non-smoker)
* region_southeast – Region indicator
* bmi_category_Obese – Obesity indicator (1 = BMI ≥ 30)

Target Variable:

* charges – Medical insurance charges

---

## Technologies Used

* Python
* Pandas & NumPy – Data handling
* Matplotlib & Seaborn – Visualization
* Scikit-learn – Machine learning
* Streamlit – Web application
* Joblib – Model persistence

---

## Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering (BMI categories, encoding)
3. Statistical Analysis (Chi-square test)
4. Train-test split
5. Model training using Random Forest Regressor
6. Model evaluation using R², MAE, RMSE
7. Feature importance analysis
8. Model deployment with Streamlit

---

## Model Performance

* Random Forest showed strong predictive performance
* Smoking status was the most influential feature
* Predictions closely matched actual insurance charges

---

## How to Run the Project

### Clone the Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Create and Activate Environment (Optional)

```
conda create -n data_env python=3.10
conda activate data_env
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Streamlit App

```
streamlit run app.py
```

---

## Streamlit App Functionality

* User-friendly input controls
* Real-time prediction of insurance charges
* Clear display of predicted results

---

## Project Structure

```
├── app.py                # Streamlit application
├── rf_model.pkl          # Trained Random Forest model
├── scaler.pkl            # Scaler (optional)
├── notebook.ipynb        # Data analysis and model training
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Conclusion

This project demonstrates an end-to-end machine learning pipeline from data preprocessing to model deployment. The results highlight the strong impact of smoking, age, and BMI on medical insurance charges.

---

## Contact

Author: Qasim Hussain

Email : qasim.ai.dev@gmail.com
