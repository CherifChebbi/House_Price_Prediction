# House Price Prediction Using Machine Learning 🏠

Welcome to the **House Price Prediction** project! This project uses machine learning techniques to predict the price of a house based on various features like size, location, and condition. By utilizing a range of regression models, the goal is to create an efficient and reliable way to estimate house prices.
![Capture d'écran 2024-12-25 162754](https://github.com/user-attachments/assets/d051eabf-3ee4-4141-a860-91cda4477409)


---

## Introduction

Predicting house prices can be a complex task involving multiple factors such as location, size, condition, and more. In this project, we utilize machine learning to predict house prices based on a given dataset. The model can help real estate agents, buyers, and sellers estimate the value of a house more accurately.

---

## Dataset Overview 📊

The dataset used in this project contains the following features:

| **Feature**       | **Description**                                                                 |
|-------------------|---------------------------------------------------------------------------------|
| **Id**            | Record identifier                                                              |
| **MSSubClass**    | Type of dwelling involved in the sale                                           |
| **MSZoning**      | General zoning classification of the sale                                       |
| **LotArea**       | Lot size (square feet)                                                          |
| **LotConfig**     | Configuration of the lot                                                        |
| **BldgType**      | Type of dwelling                                                               |
| **OverallCond**   | Overall condition of the house (rated numerically)                              |
| **YearBuilt**     | Original construction year                                                     |
| **YearRemodAdd**  | Remodel year (or construction year if no remodel)                               |
| **Exterior1st**   | Exterior covering on the house                                                  |
| **BsmtFinSF2**    | Type 2 finished square feet                                                     |
| **TotalBsmtSF**   | Total basement square footage                                                   |
| **SalePrice**     | Target variable representing the sale price                                     |

---

## Project Objective

The objective of this project is to develop a machine learning model that can accurately predict house prices based on the provided features. The model will be evaluated based on performance metrics like **Mean Squared Error (MSE)** and **R²** to determine its accuracy.

---

## Steps Involved

### 1. Data Preprocessing 🧹
- Inspecting and cleaning the dataset
- Handling missing values
- Encoding categorical variables
- Feature scaling using standardization

### 2. Exploratory Data Analysis (EDA) 🔍
- Visualizing the distributions of categorical and numerical variables
- Analyzing correlations between features
- Understanding feature relationships through various plots

### 3. Model Building and Evaluation 🏗️
- Training multiple models: Linear Regression, Random Forest, XGBoost, and more
- Evaluating model performance using metrics like MSE and R²
- Comparing the results to select the best-performing model

### 4. Model Interpretation 🧐
- Analyzing feature importance from the Random Forest model
- Gaining insights into which features are driving the predictions

### 5. Model Deployment 🚀
- Saving the trained model using `joblib` for future use
- Deploying the model for real-time predictions (e.g., using Streamlit)

---
