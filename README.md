# 🔥 Calories Burnt Prediction using Machine Learning

A Machine Learning project that predicts the number of calories burned during exercise using user and workout attributes. The project follows the complete ML pipeline, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and saving the trained model for future use.

---

## 📌 Project Overview

Estimating calories burned during physical activity is useful for tracking fitness progress and maintaining a healthy lifestyle. This project uses **XGBoost Regressor** to predict calories burned based on personal and exercise-related information.

The model is trained on historical exercise data and evaluated using regression metrics to measure prediction accuracy.

---

## 🎯 Problem Statement

Build a Machine Learning model that accurately predicts the number of calories burned using the following features:

- Gender
- Age
- Height
- Weight
- Exercise Duration
- Heart Rate
- Body Temperature

---

## 📊 Dataset

The project uses two datasets:

- **exercise.csv**
- **calories.csv**

These datasets are merged using the **User_ID** column to create the final dataset used for training.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Saving | Pickle |
| Development Environment | Jupyter Notebook |

---

## 📂 Project Workflow

```text
Load Dataset
      │
      ▼
Merge Datasets
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Correlation Analysis
      │
      ▼
Feature Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Train XGBoost Regressor
      │
      ▼
Model Evaluation
(MAE & R² Score)
      │
      ▼
Feature Importance Analysis
      │
      ▼
Actual vs Predicted Visualization
      │
      ▼
Predict Calories for New User
      │
      ▼
Save Trained Model (.pkl)
```

---

## ✨ Features

- ✅ Data Cleaning and Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Distribution Analysis
- ✅ Correlation Heatmap
- ✅ Feature Encoding
- ✅ Train-Test Split
- ✅ XGBoost Regression Model
- ✅ Mean Absolute Error (MAE)
- ✅ R² Score Evaluation
- ✅ Feature Importance Analysis
- ✅ Actual vs Predicted Visualization
- ✅ Prediction on New User Data
- ✅ Model Saving using Pickle

---

## 📈 Model Evaluation

The model performance is evaluated using:

- **Mean Absolute Error (MAE)**
- **R² Score**

These metrics help measure the prediction accuracy and overall performance of the regression model.

---

## 📁 Repository Structure

```text
Calories-Burnt-Prediction-ML
│
├── data
│   ├── calories.csv
│   └── exercise.csv
│
├── notebooks
│   └── Calories_Burnt_Prediction.ipynb
│
├── models
│   └── calories_model.pkl
│
├── images
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Calories-Burnt-Prediction-ML.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

```text
Calories_Burnt_Prediction.ipynb
```

Run all cells in sequence.

---

## 💡 Sample Prediction

Input:

| Feature | Value |
|---------|------:|
| Gender | Male |
| Age | 24 |
| Height | 175 cm |
| Weight | 70 kg |
| Duration | 30 min |
| Heart Rate | 120 bpm |
| Body Temperature | 40 °C |

Output:

```text
Predicted Calories Burned ≈ 245 Calories
```

---

## 📷 Project Screenshots

Add screenshots here after uploading.

- Dataset Overview
- Distribution Plot
- Correlation Heatmap
- Feature Importance
- Actual vs Predicted Graph

---

## 🔮 Future Improvements

- Deploy the model using Streamlit
- Hyperparameter Optimization
- Use larger real-world datasets
- Build a Fitness Recommendation System
- Integrate with wearable fitness devices

---

## 👨‍💻 Author

**Shubh Chak**

Aspiring Data Analyst | Machine Learning Enthusiast

---

## ⭐ If you found this project useful, consider giving it a Star!
