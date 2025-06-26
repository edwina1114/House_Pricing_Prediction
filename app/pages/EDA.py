import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.access import get_image
from utils.eda_plot import cor_plot, car_plot

st.header("📊 Exploratory Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Target Variable", "⚙️ Missing Values", "📊 Correlation", "🔢 Cardinality"]
)

with tab1:
    st.subheader("🎯 Target Variable - SalePrice")

    st.markdown(
        """
        The distribution of `SalePrice` is **right-skewed**, which can distort linear model assumptions.  
        👉 We apply a log transformation using `np.log1p()` to normalize the data:
        
        - 📉 Reduces skewness  
        - 🚫 Minimizes the impact of extreme values (e.g., luxury houses)
        """
    )

    st.image(
        get_image("EDA", "target_distribution_log.png"),
        caption="Log-Transformed Distribution of SalePrice",
        use_column_width=True,
    )

with tab2:
    st.subheader("⚙️ Handling Features with Missing Values")

    st.markdown(
        """
        This figure shows **box plots (left)** and **count plots (right)** for categorical features with missing values.  
        These visualizations guided customized preprocessing strategies for each feature group.
        """
    )

    st.markdown("#### ✅ 1. Features with Quality Levels")
    st.markdown(
        """
        - **Strategy**: Impute missing values, apply ordinal encoding, add binary indicators  
        - **Features**: `PoolQC`, `MasVnrType`, `FireplaceQu`

        **Actions**:
        - Impute with `"None"`  
        - Apply **ordinal encoding**  
        - Add indicators: `HasPool`, `HasMasonryVeneer`, `HasFireplace`
        """
    )

    st.divider()

    st.markdown("#### ✅ 2. Features with Sparse Categories")
    st.markdown(
        """
        - **Strategy**: Impute missing values + add binary indicators only  
        - **Features**: `MiscFeature`, `Alley`, `Fence`

        **Actions**:
        - Impute with `"None"`  
        - Add indicators: `HasMiscFeature`, `HasAlley`, `HasFence`
        """
    )

    st.divider()

    st.markdown("#### ✅ 3. Low-Missing Features")
    st.markdown(
        """
        - **Numerical features** → Impute using **mean**  
        - **Categorical features** → Impute using **most frequent value**
        """
    )

    st.divider()

    st.image(
        get_image("EDA", "box_count_missing_features.png"),
        caption="Box and Count Plots of Features with Missing Values",
        use_column_width=True,
    )

with tab3:
    st.subheader("📊 Correlation with SalePrice")

    st.markdown(
        """
        The following bar chart shows the top 20 features with the strongest correlation (positive or negative) to `SalePrice`.  
        Notably, house size and quality stand out as key drivers.
        """
    )

    st.plotly_chart(cor_plot(), use_container_width=True)


with tab4:
    st.subheader("🔢 Categorical Feature Cardinality")

    st.markdown(
        """
        High-cardinality categorical columns can affect model performance and encoding decisions.
        - `Neighborhood`, `Exterior2nd`, `Exterior1st` → **Frequency Encoding**  
        - Others → **One-Hot Encoding**
        """
    )

    st.plotly_chart(car_plot(), use_container_width=True)
