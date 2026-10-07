# Bank Marketing Prediction Using Logistic Regression

This repository/Colab notebook demonstrates an end-to-end Machine Learning pipeline using **Logistic Regression** to predict whether a customer will subscribe to a term deposit (binary classification: `yes` / `no`). The dataset used is the standard Bank Marketing dataset (`bank-additional-full.csv`).

---

## 📋 Overview of the Pipeline

The notebook walks through standard data science lifecycle steps:

1. **Data Loading & Inspection**: Importing essential libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`) and loading the dataset containing 41,199 rows and 21 columns.
2. **Data Preprocessing**:
   - Handling missing values (`dropna`).
   - Removing duplicate records.
   - Outlier detection and treatment using the Interquartile Range (IQR) method on skewed continuous variables (`duration`, `campaign`).
3. **Target Encoding**: Transforming the binary target variable `y` (`yes`/`no`) into numeric values (`1`/`0`).
4. **Label Encoding**: Converting all categorical object columns into numerical representations using scikit-learn's `LabelEncoder`.
5. **Multicollinearity Check (VIF)**: Calculating Variance Inflation Factors (VIF) to detect and remove highly collinear features (`euribor3m`, `emp.var.rate`) to stabilize the model.
6. **Model Training & Evaluation**:
   - Splitting the data into training and testing sets ($80\%$ train, $20\%$ test).
   - Fitting a **Logistic Regression** model using scikit-learn.
   - Evaluating model performance using **Accuracy Score**, achieving approximately **$93.1\%$ accuracy**.

---

## 🛠️ Requirements & Dependencies

To run this notebook successfully, ensure you have the following Python libraries installed:

* Python 3.x
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `statsmodels`

---

## 🚀 Key Steps & Code Snippets

### 1. Data Loading & Cleaning
python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ML CSV/bank-additional-full.csv", sep=';')

# Drop nulls and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


### 2. Feature Engineering & VIF Reduction

Features with high collinearity (such as `euribor3m` and `emp.var.rate`) were iteratively dropped based on high VIF scores to avoid multi-collinearity issues.

### 3. Model Training and Prediction

python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Feature and Target Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=124)

# Fit Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict and Evaluate
y_pred = model.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

---

## 📊 Results

* **Final Model**: Logistic Regression
* **Test Accuracy**: $\approx 93.1\%$

---

## 💡 How to Use

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/1ygg68F4lEqnSWDAgF7wLMVuDy-LAHZVR).
2. Ensure your Google Drive is mounted and the dataset path (`/content/drive/MyDrive/Colab Notebooks/ML CSV/bank-additional-full.csv`) is correctly configured.
3. Run all cells sequentially to reproduce the preprocessing steps, VIF analysis, and final model evaluation.
