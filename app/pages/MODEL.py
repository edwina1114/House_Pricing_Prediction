import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.access import get_image
from utils.model_plot import (
    create_comparison_plots,
    plot_before_after_performance,
    plot_top_features,
)

st.header("🧪 Modeling")

tab1, tab2, tab3 = st.tabs(
    [
        "⚙️ Modeling Pipeline",
        "📈 Model Comparison",
        "🧠 Feature Selection Insights",
    ]
)

with tab1:
    st.subheader("⚙️ Modeling Pipeline")

    st.markdown(
        """

        This pipeline processes different feature groups using customized transformers:

        1. **Numerical Features**  
           ➤ Imputed using **mean** → Scaled with **StandardScaler**

        2. **Categorical Features (Low Cardinality)**  
           ➤ Imputed using **most frequent** → Encoded with **OneHotEncoder**

        3. **Quality-Based Features** (`PoolQC`, `FireplaceQu`, etc.)  
           ➤ Imputed with `"None"`  
           ➤ Transformed into **ordinal values** (e.g., None < TA < Gd < Ex)  
           ➤ **Binary presence indicators** also added

        4. **Presence-Only Features** (`Alley`, `Fence`, `MiscFeature`, etc.)  
           ➤ Imputed with `"None"`  
           ➤ Encoded only as **binary indicators** (e.g., `HasAlley`)

        5. **High-Cardinality Categorical Features** (`Neighborhood`, `Exterior1st`, `Exterior2nd`)  
           ➤ Transformed using **Frequency Encoding** (based on relative frequency)

        """
    )

    st.image(
        get_image("MODEL", "pipeline.png"),
        use_container_width=True,
    )

with tab2:
    st.subheader("📈 Model Overview")
    st.markdown(
        """
        - **Five models were evaluated with 5-Fold Cross Validation:**  
          Linear Regression, Ridge, Lasso, Random Forest, XGBoost

        - **Evaluation Metrics:**  
          - MAE (Mean Absolute Error)  
          - R² (Coefficient of Determination)  
          - Overfitting Gap (Train vs Validation R²)

        - ✅ **Lasso** showed the best trade-off between accuracy and generalization.
        """
    )

    st.plotly_chart(create_comparison_plots(), use_container_width=True)

    st.subheader("📊 Summary & Insights")
    st.markdown(
        """
        #### 🔍 Model Selection: Balancing Accuracy and Generalization
        - **Linear Regression** had top accuracy (MAE: 16,333; R²: 0.920) but poor generalization with a **36% overfitting gap**.
        - A **weighted metric = MAE × (1 + Overfitting Gap%)** was used for fair comparison.

        #### ✅ Best Overall: Lasso Regression
        - Strong MAE and **lowest overfitting gap (~12%)**.
        - Includes **automatic feature selection** for simplicity and robustness.

        #### ✅ Linear > Tree-Based Models
        - **Linear, Ridge, Lasso** outperformed **Random Forest, XGBoost**
        - Implies data has **linear tendencies** and small size limits tree models (~1,460 rows)

        #### ⚠️ Weakness in High-Priced Homes
        - All models had **larger residuals on expensive homes**
        - **Future work:** Stratified modeling or feature engineering for luxury homes
        """
    )

    st.subheader("🔍 Residuals vs Predicted Values")
    st.image(get_image("MODEL", "residuals_vs_predicted_comparison_across_models.png"))


with tab3:
    st.subheader("🧠 Lasso Feature Selection Summary")
    st.markdown(
        """
        | Metric | Value |
        |--------|-------|
        | Final Model | Lasso Regression |
        | Feature Selector | Lasso + SelectFromModel |
        | Features Retained | 76 / 220 (34.5%) |
        | R² (Validation) | 0.9120 |
        | MAE (Validation) | 17,139 |
        | Performance Retention | 100% |
        | Compression Ratio | 2.9× |
        """
    )

    st.plotly_chart(plot_before_after_performance(), use_container_width=True)

    st.subheader("🧠 Top 20 Features from Lasso")
    st.plotly_chart(plot_top_features(), use_container_width=True)

    st.markdown(
        """
        - `RoofMatl_ClyTile` has the strongest negative impact on price.  
          ➤ May reflect outdated design or poor neighborhood quality.

        - `GrLivArea` and `OverallQual` are top positive predictors.  
          ➤ Larger, higher-quality homes command better prices.

        - Most numerical features (e.g., `GarageCars`, `YearBuilt`) show positive influence.  
          ➤ Suggests that **size and recency** are key price drivers.
        """
    )
