import pandas as pd
import plotly.express as px
from utils.access import get_data


def cor_plot():
    correlation = pd.read_csv(get_data("correlation.csv"))
    top20 = correlation.reindex(
        correlation["Correlation"].abs().sort_values(ascending=False).head(20).index
    )
    colors = ["steelblue" if x > 0 else "lightcoral" for x in top20["Correlation"]]

    fig = px.bar(
        x=top20["Correlation"],
        y=top20["Features"],
        orientation="h",
        color=top20["Correlation"],
        color_continuous_scale=[[0.0, "lightblue"], [0.5, "dodgerblue"], [1.0, "navy"]],
        title="Top 20 Features by Correlation Strength with SalePrice",
    )

    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Correlation: %{x:.4f}<extra></extra>",
    )

    fig.update_layout(
        width=900,
        height=750,
        xaxis_title="Correlation with SalePrice",
        yaxis_title="Features",
        margin=dict(l=100, r=80, t=80, b=50),
        coloraxis_colorbar=dict(title="Correlation<br>Strength"),
    )
    return fig


def car_plot():
    cat_cardinality = pd.read_csv(get_data("cardinality.csv"))
    top_categorical = cat_cardinality.sort_values("Cardinality", ascending=False).head(
        20
    )

    fig = px.bar(
        y=top_categorical["Feature"],
        x=top_categorical["Cardinality"],
        orientation="h",
        color=top_categorical["Cardinality"],
        color_continuous_scale=[
            [0.0, "#c3eec9"],  # seafoam
            [0.5, "#5cb270"],  # fresh green
            [1.0, "#184e3f"],  # pine green
        ],
        title="Top 20 Categorical Features by Cardinality",
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside", textfont_size=12)

    fig.update_layout(
        width=900,
        height=750,
        xaxis_title="Number of Unique Categories",
        yaxis_title="Categorical Features (Ranked by Cardinality)",
        margin=dict(l=120, r=80, t=80, b=50),
        legend=dict(title="Cardinality Level", x=1.02, y=1),
        coloraxis_colorbar=dict(title="Cardinality<br>Level"),
    )
    return fig
