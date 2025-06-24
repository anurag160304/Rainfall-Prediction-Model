

## 🌧️ Rainfall Prediction Project

This project implements various machine learning models to predict rainfall using the **Australian weather dataset**.

---

## 📌 Project Overview

The goal of this project is to explore and compare different classification algorithms for predicting **whether it will rain tomorrow**. The process involves:

* Loading and inspecting the data
* Preprocessing and cleaning
* Feature engineering and selection
* Training multiple models
* Evaluating and comparing model performance
* Visualizing the results

---

## 📊 Dataset

The analysis is based on the **`weatherAUS.csv`** dataset, which contains **daily weather observations** from various locations across Australia.

---

## 🧪 Methodology

### 1. Data Loading & Initial Inspection

* Loaded the dataset into a pandas DataFrame.
* Explored its structure and basic statistics.

### 2. Data Cleaning & Preprocessing

* Converted categorical variables (`RainToday`, `RainTomorrow`) to numerical format.
* Handled missing values:

  * **Categorical**: Imputed using mode.
  * **Numerical**: Imputed using **IterativeImputer**.
* Detected and removed outliers using the **IQR method**.

### 3. Feature Engineering & Selection

* Standardized numerical features using **MinMaxScaler**.
* Selected relevant features using:

  * **Chi-Square test** (filter method)
  * **Random Forest** (wrapper method)

### 4. Model Development & Evaluation

* Split data into training and test sets.
* Trained multiple classification models:

  * Logistic Regression
  * Decision Tree
  * Neural Network
  * Random Forest
  * LightGBM (`LGBMClassifier`)
  * CatBoost (`CatBoostClassifier`)
  * XGBoost (`XGBClassifier`)
* Evaluated using:

  * Accuracy
  * ROC AUC
  * Cohen’s Kappa
  * Confusion Matrices and ROC curves

### 5. Model Comparison

* Compared all models based on performance and training time.
* Identified top performers and analyzed trade-offs (e.g., speed vs. accuracy).

### 6. Prediction with New Data

* Demonstrated predictions on new, simulated input data using trained models.

---

## 🤖 Models Implemented

* ✅ Logistic Regression
* ✅ Decision Tree
* ✅ Neural Network
* ✅ Random Forest
* ✅ LightGBM (`LGBMClassifier`)
* ✅ CatBoost (`CatBoostClassifier`)
* ✅ XGBoost (`XGBClassifier`)

---

## 📈 Key Findings

* The **Neural Network**, **XGBoost**, and **CatBoost** models achieved the **highest overall performance**.
* Final model selection may depend on resource availability and speed requirements.

---

## 📁 Repository Structure

```
📄 weatherAUS.csv             # Dataset file  
📓 Rain_Prediction_Model.ipynb   # Jupyter notebook with all code  
📘 README.md                  # Project documentation (this file)
```

---

## 🚀 Getting Started

To run this project on your local machine:

1. Clone this repository:

```bash
git clone https://github.com/anurag160304/Rain_Prediction_model.git
cd /anurag160304/Rain_Prediction_model
```

2. Install the required libraries:

```bash
!pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost xgboost mlxtend
```

> Or manually install key libraries like: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, etc.

3. Open the notebook:

```bash
jupyter notebook Rain_prediction_Model.ipynb
```


