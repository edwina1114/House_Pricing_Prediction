import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.access import get_data


def create_comparison_plots():
    df_results = pd.read_csv(get_data("df_results.csv"))

    colors = df_results["color"].tolist()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Validation MAE Comparison",
            "Validation R² Comparison",
            "Overfitting Analysis",
            "Performance vs Training Time",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.2,
    )

    # 1. MAE Comparison
    fig.add_trace(
        go.Bar(
            x=df_results["Model"],
            y=df_results["Valid_MAE"],
            marker_color=colors,
            name="Valid MAE",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. R² Comparison
    fig.add_trace(
        go.Bar(
            x=df_results["Model"],
            y=df_results["Valid_R2"],
            marker_color=colors,
            name="Valid R²",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Overfitting Analysis
    fig.add_trace(
        go.Bar(
            x=df_results["Model"],
            y=df_results["Overfitting_Gap_%"],
            marker_color=colors,
            name="Overfitting Gap",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # threshold lines for overfitting
    fig.add_hline(
        y=15,
        line_dash="dash",
        line_color="orange",
        annotation_text="Acceptable (15%)",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=25,
        line_dash="dash",
        line_color="red",
        annotation_text="Problematic (25%)",
        row=2,
        col=1,
    )

    # 4. Performance vs Training Time
    fig.add_trace(
        go.Scatter(
            x=df_results["Training_Time_s"],
            y=df_results["Valid_MAE"],
            mode="markers",  # 只顯示點，不顯示文字
            marker=dict(color=colors, size=15, opacity=0.8),  # 稍微增大點的大小
            text=df_results["Model"],  # 模型名稱
            customdata=df_results[
                ["Train_R2", "Valid_R2", "Overfitting_Gap_%"]
            ],  # 額外資訊
            hovertemplate="<b>%{text}</b><br>"
            + "Training Time: %{x:.1f} seconds<br>"
            + "Validation MAE: %{y:.0f}<br>"
            + "Train R²: %{customdata[0]:.3f}<br>"
            + "Valid R²: %{customdata[1]:.3f}<br>"
            + "Overfitting Gap: %{customdata[2]:.1f}%<br>"
            + "<extra></extra>",  # 移除預設的 trace 名稱框
            name="Models",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=850, width=1000, title_text="Model Comparison Results", showlegend=False
    )

    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=1)

    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=1, col=2)

    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Overfitting Gap (%)", row=2, col=1)

    fig.update_xaxes(title_text="Training Time (seconds)", row=2, col=2)
    fig.update_yaxes(title_text="Validation MAE", row=2, col=2)
    return fig


def plot_before_after_performance():
    with open(get_data("final_model_info.json"), "r") as f:
        final_model_info_json = json.load(f)

    mae_before = final_model_info_json["original_mae"]
    mae_after = final_model_info_json["selected_mae"]
    r2_before = final_model_info_json["original_r2"]
    r2_after = final_model_info_json["selected_r2"]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("MAE Before vs After", "R² Before vs After")
    )

    # MAE Plot
    fig.add_trace(
        go.Bar(
            x=["Before", "After"],
            y=[mae_before, mae_after],
            marker_color=["#8ecae6", "#219ebc"],
            text=[f"{mae_before:.0f}", f"{mae_after:.0f}"],
            textposition="auto",
            name="MAE",
        ),
        row=1,
        col=1,
    )

    # R² Plot
    fig.add_trace(
        go.Bar(
            x=["Before", "After"],
            y=[r2_before, r2_after],
            marker_color=["#ffb703", "#fb8500"],
            text=[f"{r2_before:.4f}", f"{r2_after:.4f}"],
            textposition="auto",
            name="R²",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        width=1000,
        title_text="Model Performance Before vs After Feature Selection",
        showlegend=False,
    )

    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)

    return fig


def plot_top_features():
    top_n = 20

    with open(get_data("lasso_top_feature_info.json")) as f:
        coef_info = json.load(f)

    coefficients = np.array(coef_info["coef"])
    feature_names = coef_info["feature_names"]

    selected_indices = np.where(coefficients != 0)[0]
    selected_coefs = coefficients[selected_indices]

    importance_order = np.argsort(-np.abs(selected_coefs))
    top_indices = importance_order[:top_n]
    top_coefs = selected_coefs[top_indices]

    if feature_names is not None:
        try:
            top_feature_names = [
                feature_names[selected_indices[idx]] for idx in top_indices
            ]
        except IndexError:
            top_feature_names = [
                f"Feature_{selected_indices[idx]}" for idx in top_indices
            ]
    else:
        top_feature_names = [f"Feature_{selected_indices[idx]}" for idx in top_indices]

    colors = ["steelblue" if coef > 0 else "lightcoral" for coef in top_coefs]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=top_feature_names,
            x=top_coefs,
            orientation="h",
            marker=dict(color=colors, opacity=0.7, line=dict(color="black", width=0.5)),
            text=[f"{coef:.3f}" for coef in top_coefs],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>"
            + "Coefficient: %{x:.4f}<br>"
            + "<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.3, line_width=1)

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Features",
            font=dict(size=16, family="Arial Black"),
            xanchor="center",
            x=0.1,
        ),
        xaxis=dict(
            title="Lasso Coefficient",
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(title="Features", tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=200, r=50, t=80, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
