import streamlit as st

st.set_page_config(page_title="🏠 House Prices App", layout="centered")

st.title("🏡 House Prices Prediction Project")

st.caption("📌 Navigate through the sidebar to explore each section.")

st.markdown(
    """
    This dashboard demonstrates a complete machine learning pipeline to predict housing prices 
    using the **Ames Housing Dataset**. It walks through key steps from:
    
    - 📊 **Exploratory Data Analysis (EDA)**  
    - ⚙️ **Model Training & Evaluation**  
    - 🧠 **Feature Selection & Interpretation**

    ---
    📂 **Dataset Source**:  
    [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/home-data-for-ml-course)

    
    """
)
