# Diabetes Prediction using Machine Learning

## Project Overview

This project focuses on predicting whether a patient is likely to have diabetes based on medical diagnostic measurements. The goal is to build a machine learning model that can assist in early detection of diabetes using supervised learning algorithms.

The dataset used contains several health-related features such as glucose level, blood pressure, insulin level, BMI, and age. By analyzing these attributes, the model learns patterns that help predict the likelihood of diabetes.

---

## Dataset Description

The dataset consists of **768 patient records** and **9 features**.

### Features

* Pregnancies – Number of pregnancies
* Glucose – Plasma glucose concentration
* BloodPressure – Diastolic blood pressure (mm Hg)
* SkinThickness – Triceps skin fold thickness (mm)
* Insulin – 2-Hour serum insulin (mu U/ml)
* BMI – Body Mass Index
* DiabetesPedigreeFunction – Diabetes genetic influence
* Age – Age of the patient
* Outcome – Diabetes result

  * 0 → Non-Diabetic
  * 1 → Diabetic

Total diabetic patients in the dataset: **268**

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* Google Colab

---

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas and basic information such as head, tail, info, and statistical description is analyzed.

### 2. Data Exploration

Exploratory Data Analysis (EDA) is performed to understand patterns in the dataset.

Visualizations include:

* Scatter plot (BMI vs Age)
* Histogram of Diabetes Pedigree Function
* Bar chart between Blood Pressure and Insulin
* Correlation Heatmap
* Boxplots for feature analysis

### 3. Data Cleaning

Some columns contain **zero values which are not medically valid**, such as:

* Glucose
* BloodPressure
* SkinThickness
* Insulin

These values are replaced with **NaN** and rows with missing values are removed.

### 4. Outlier Handling

Outliers are detected using the **IQR method** and handled through **capping** to reduce their effect on model performance.

### 5. Feature Selection

The target variable **Outcome** is separated from the feature set to create:

* X → Input features
* y → Target variable

### 6. Train-Test Split

The dataset is split into training and testing sets:

* 80% Training Data
* 20% Testing Data

---

## Machine Learning Models Used

### 1. Logistic Regression

Logistic Regression is used as a baseline supervised learning classification model.

Results:

* Accuracy: **74.67%**
* Precision: **0.64**
* Recall: **0.67**
* F1 Score: **0.65**

A confusion matrix is used to evaluate model predictions.

---

### 2. Decision Tree Classifier

Decision Tree is another supervised learning model used for classification.

Results:

* Accuracy: **75.32%**
* Precision: **0.63**
* Recall: **0.75**
* F1 Score: **0.68**

Decision Tree slightly outperformed Logistic Regression in this experiment.

---

## Model Evaluation Metrics

The following metrics were used to evaluate the models:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics help in understanding the model's performance for classification problems.

---

## Results Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 74.67%   | 0.64      | 0.67   | 0.65     |
| Decision Tree       | 75.32%   | 0.63      | 0.75   | 0.68     |

The Decision Tree model provided slightly better performance compared to Logistic Regression.

---

## Conclusion

This project demonstrates how machine learning can be applied in healthcare to predict diseases such as diabetes. Through data preprocessing, exploratory data analysis, and model training, we built predictive models that can assist in early detection.

Although the accuracy is moderate, the project highlights the importance of data preprocessing and feature engineering in improving machine learning performance.

---

## Future Improvements

* Use advanced models such as Random Forest or XGBoost
* Perform hyperparameter tuning
* Apply feature scaling and normalization
* Deploy the model using a web framework
* Integrate the model with a Streamlit application

---

## Author

Meeta
B.Tech CSE (AI & ML)
Lovely Professional University

