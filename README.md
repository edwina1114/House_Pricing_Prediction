# 🏡 House Prices Prediction Project

## 📌 Overview

This project aims to predict house prices using the Ames Housing dataset. It includes comprehensive steps from **exploratory data analysis (EDA)** to **model selection** and **feature optimization**.

📂 Dataset Source:  
[Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/home-data-for-ml-course)

The process is documented in two Jupyter Notebooks:

- [01_eda_insights.ipynb](notebooks/01_eda_insights.ipynb): EDA and feature understanding  
- [02_complete_modeling.ipynb](notebooks/02_complete_modeling.ipynb): Modeling, comparison, and final model refinement

---
## 🚀 Live Demo
🖥️ Explore the interactive dashboard and in-depth insights:
[🔗 Streamlit Live Demo](https://housepricingprediction-xxjmo3k84wor4zjrgrjfhz.streamlit.app/)

The Streamlit app includes:

- 🎯 EDA summaries and visualizations

- 📈 Model performance comparisons

- 🧠 Feature selection insights

- 📊 Dynamic charts and metrics
  
---

## 🔍 Exploratory Data Analysis Highlights

Key insights from [01_eda_insights.ipynb](notebooks/01_eda_insights.ipynb):

- 🔄 **Target Transformation**: `SalePrice` was log-transformed to reduce right skew and improve model fit  
- 🧱 **Missing Value Handling**: Applied customized strategies based on feature semantics (e.g., quality vs presence)  
- 🧮 **Categorical Cardinality**: High-cardinality features (e.g., `Neighborhood`) were frequency-encoded; others used one-hot encoding  
- 📊 **Feature Correlation**: `OverallQual`, `GrLivArea`, and `GarageCars` showed strongest positive correlations with price  

> 📎 Visualizations available in [images/EDA/](images/EDA/) include:  
> - [Top Correlated Features](images/EDA/correlation_with_saleprice.png) 
> - [Categorical Features Cardinality](images/EDA/cardinality_of_categorical_features.png)
> - [Missing Value Box/Count Plots](images/EDA/box_count_missing_features.png)  
> - [Log-Transformed Target Distribution](images/EDA/target_distribution_log.png)

---

## 🧪 Modeling Pipeline

📌 Key steps from `02_complete_modeling.ipynb`:

### 🧹 Data Preprocessing
- Ordinal + Binary indicators for quality-related features (`PoolQC`, `FireplaceQu`)
- Binary indicators for sparse features (`Alley`, `Fence`)
- Frequency encoding for high-cardinality features (`Neighborhood`, `Exterior1st`)
- Combined all pipelines via `ColumnTransformer`

---

## 📈 Modeling & Evaluation

🧪 Compared models:  
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest  
- XGBoost

📏 Metrics used:  
- **MAE** (Mean Absolute Error)  
- **R² Score**  
- **Overfitting Gap (%)** = Train R² − Validation R²

---

## ✅ Final Model: Lasso Regression

- 🏆 Balanced: MAE ≈ 17,139, R² ≈ 0.912, gap ≈ 12%  
- ⚙️ Built-in feature selection improved generalization

---

## 🔎 Feature Selection with Lasso

- ✅ Retained **76 / 220 features (34.5%)**  
- 📉 Achieved **2.9× compression ratio**  
- 🔧 No drop in performance

---

## 📊 Performance Before vs After Feature Selection

| Metric            | Before | After |
|-------------------|--------|-------|
| MAE               | 17,139 | 17,139 |
| R² Score          | 0.9120 | 0.9120 |
| Retained Features | 220    | 76     |

---

## 🛠️ Tech Stack

- **Programming**: Python  
- **Libraries**:  
  - `pandas`, `numpy` for data processing  
  - `matplotlib`, `seaborn`, `plotly` for visualization  
  - `scikit-learn`, `xgboost` for modeling  
  - `streamlit` for app deployment

---

## 📁 Project Structure
```
HOUSE_PRICING_PREDICTION/
├── app/                       # Streamlit app and supporting utils
│   ├── pages/                 # Streamlit tab pages
│   ├── utils/                 # Custom Functions
│   ├── data/                  # Intermediate CSVs for visualization
│   └── Home.py                # Main dashboard entry
├── data/ # Raw training and test datasets
│ ├── train.csv
│ └── test.csv
├── images/ # All result plots and visualizations
│ ├── EDA/ # Visualizations from exploratory analysis
│ └── MODEL/ # Model performance plots
├── notebooks/ 
│ ├── 01_eda_insights.ipynb # EDA and feature analysis
│ └── 02_complete_modeling.ipynb # Modeling and evaluation pipeline
└── README.md # Project overview and documentation
```

---

## 🚀 Streamlit - How to Run Locally

### ▶️ Pip 
```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

