"""Single-file Streamlit app: Manufacturing Defect & Quality Control Analytics.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import importlib.util

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
if SKLEARN_AVAILABLE:
    from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Manufacturing Defect & Quality Control Analytics", layout="wide")
px.defaults.template = "plotly_dark"

st.markdown(
    """
    <style>
        .main {
            background: radial-gradient(circle at 20% 20%, rgba(29,78,216,0.35) 0%, rgba(8,12,30,0.95) 38%, #04070f 100%);
            color: #f5f7ff;
        }
        .stMetric {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 0 20px rgba(58, 109, 240, 0.25);
        }
        .glow-title {
            font-size: 2.35rem;
            font-weight: 900;
            background: linear-gradient(90deg, #64e3ff, #9d8bff, #ff7edb, #64e3ff);
            background-size: 220% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 26px rgba(100, 227, 255, 0.55), 0 0 45px rgba(157, 139, 255, 0.35);
            animation: pulse 2.8s infinite, shimmer 4s linear infinite;
        }
        @keyframes pulse {
            0% { filter: brightness(1); }
            50% { filter: brightness(1.25); }
            100% { filter: brightness(1); }
        }
        @keyframes shimmer {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }
        .insight-card {
            background: rgba(255,255,255,0.08);
            border-left: 4px solid #64e3ff;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 8px 0;
        }
        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 12px;
            font-weight: 700;
            margin-right: 8px;
            animation: floatIn 1.2s ease;
        }
        .status-green {
            background: rgba(34,197,94,0.2);
            border: 1px solid rgba(34,197,94,0.55);
            color: #86efac;
        }
        .status-red {
            background: rgba(239,68,68,0.2);
            border: 1px solid rgba(239,68,68,0.55);
            color: #fca5a5;
        }
        .pulse-soft {
            animation: pulseSoft 2.2s infinite;
        }
        @keyframes pulseSoft {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.04); opacity: 0.82; }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes floatIn {
            from { transform: translateY(8px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def generate_synthetic_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    suppliers = [f"S{i:02d}" for i in range(1, 11)]
    regions = ["North", "South", "East", "West"]
    plants = ["Plant_A", "Plant_B", "Plant_C", "Plant_D", "Plant_E", "Plant_F", "Plant_G"]

    region_defect_bias = {"North": 0.010, "South": 0.015, "East": 0.020, "West": 0.012}

    supplier_ids = rng.choice(suppliers, size=rows)
    supplier_regions = rng.choice(regions, size=rows, p=[0.24, 0.30, 0.23, 0.23])
    plant_locations = rng.choice(plants, size=rows)
    date_col = rng.choice(dates, size=rows)

    production_volume = rng.integers(1000, 12000, size=rows)
    supplier_risk = {s: rng.uniform(-0.002, 0.006) for s in suppliers}

    seasonal = pd.Series(date_col).dt.month.map(
        {
            1: 0.001,
            2: 0.001,
            3: 0.0005,
            4: 0.0,
            5: -0.0005,
            6: -0.001,
            7: -0.0008,
            8: -0.0003,
            9: 0.0003,
            10: 0.0007,
            11: 0.0012,
            12: 0.0015,
        }
    ).values

    defect_probs = np.array(
        [
            np.clip(region_defect_bias[r] + supplier_risk[s] + season + rng.normal(0, 0.0025), 0.001, 0.09)
            for r, s, season in zip(supplier_regions, supplier_ids, seasonal)
        ]
    )

    defect_count = rng.binomial(production_volume, defect_probs)
    inspection_result = np.where((defect_count / production_volume) <= 0.022, "Pass", "Fail")

    df = pd.DataFrame(
        {
            "Date": date_col,
            "Supplier_ID": supplier_ids,
            "Supplier_Region": supplier_regions,
            "Plant_Location": plant_locations,
            "Production_Volume": production_volume,
            "Defect_Count": defect_count,
            "Inspection_Result": inspection_result,
        }
    )

    missing_indices = rng.choice(df.index, size=max(1, int(rows * 0.01)), replace=False)
    midpoint = len(missing_indices) // 2
    df.loc[missing_indices[:midpoint], "Plant_Location"] = np.nan
    df.loc[missing_indices[midpoint:], "Production_Volume"] = np.nan

    return df


def clean_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce")

    work_df["Plant_Location"] = work_df["Plant_Location"].fillna("Unknown")
    work_df["Production_Volume"] = work_df["Production_Volume"].fillna(work_df["Production_Volume"].median())
    work_df = work_df.dropna(subset=["Date", "Supplier_ID", "Supplier_Region", "Defect_Count"])

    work_df["Defect_Rate"] = work_df["Defect_Count"] / work_df["Production_Volume"]

    for col in ["Production_Volume", "Defect_Count", "Defect_Rate"]:
        min_val, max_val = work_df[col].min(), work_df[col].max()
        denom = max_val - min_val
        work_df[f"{col}_Normalized"] = 0.0 if denom == 0 else (work_df[col] - min_val) / denom

    work_df["Month"] = work_df["Date"].dt.to_period("M").astype(str)
    return work_df


def hypothesis_test_by_region(df: pd.DataFrame) -> tuple[float, float]:
    grouped = [grp["Defect_Rate"].values for _, grp in df.groupby("Supplier_Region") if len(grp) > 1]
    if len(grouped) < 2:
        return np.nan, np.nan
    f_stat, p_value = stats.f_oneway(*grouped)
    return float(f_stat), float(p_value)


def calculate_financial_impact(df: pd.DataFrame, unit_cost: float, defect_penalty: float, reduction_pct: float) -> dict:
    financial_df = df.copy()
    financial_df["Gross_Profit"] = financial_df["Production_Volume"] * unit_cost
    financial_df["Quality_Cost"] = financial_df["Defect_Count"] * defect_penalty
    financial_df["Improved_Quality_Cost"] = financial_df["Quality_Cost"] * (1 - reduction_pct / 100)
    financial_df["Potential_Savings"] = financial_df["Quality_Cost"] - financial_df["Improved_Quality_Cost"]

    by_plant = (
        financial_df.groupby("Plant_Location", as_index=False)[
            ["Gross_Profit", "Quality_Cost", "Improved_Quality_Cost", "Potential_Savings"]
        ]
        .sum()
        .sort_values("Quality_Cost", ascending=False)
    )

    total_quality_loss = float(financial_df["Quality_Cost"].sum())
    potential_savings = float(financial_df["Potential_Savings"].sum())
    return {
        "plant_financials": by_plant,
        "total_quality_loss": total_quality_loss,
        "potential_savings": potential_savings,
    }


def split_relevant_unwanted_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevant_mask = (
        (df["Plant_Location"] != "Unknown")
        & (df["Production_Volume"] > 0)
        & (df["Defect_Count"] >= 0)
        & (df["Defect_Rate"] <= 0.10)
    )
    relevant_df = df[relevant_mask].copy()
    unwanted_df = df[~relevant_mask].copy()
    return relevant_df, unwanted_df


@st.cache_resource
def train_risk_model(df: pd.DataFrame):
    if not SKLEARN_AVAILABLE:
        return None, []

    model_df = df[["Supplier_ID", "Supplier_Region", "Production_Volume", "Inspection_Result"]].copy()
    encoded = pd.get_dummies(model_df[["Supplier_ID", "Supplier_Region", "Production_Volume"]], drop_first=False)
    target = model_df["Inspection_Result"].eq("Fail").astype(int)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(encoded, target)
    return model, encoded.columns.tolist()


def build_supplier_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    supplier_stats = df.groupby("Supplier_ID").agg(
        Defect_Rate=("Defect_Rate", "mean"),
        Volume_Mean=("Production_Volume", "mean"),
        Volume_Std=("Production_Volume", "std"),
        Dominant_Region=("Supplier_Region", lambda x: x.mode().iloc[0]),
    )
    supplier_stats["Volume_Std"] = supplier_stats["Volume_Std"].fillna(0)
    supplier_stats["Volume_CV"] = supplier_stats["Volume_Std"] / supplier_stats["Volume_Mean"].replace(0, np.nan)
    supplier_stats["Volume_CV"] = supplier_stats["Volume_CV"].fillna(0)

    region_reliability = (
        df.assign(Fail_Flag=df["Inspection_Result"].eq("Fail").astype(int))
        .groupby("Supplier_Region")["Fail_Flag"]
        .mean()
        .apply(lambda x: 1 - x)
    )
    supplier_stats["Regional_Reliability"] = supplier_stats["Dominant_Region"].map(region_reliability).fillna(0.5)

    defect_score = (1 - supplier_stats["Defect_Rate"].rank(pct=True)) * 100
    consistency_score = (1 - supplier_stats["Volume_CV"].rank(pct=True)) * 100
    reliability_score = supplier_stats["Regional_Reliability"] * 100

    supplier_stats["Quality_Health_Score"] = (
        0.5 * defect_score + 0.3 * consistency_score + 0.2 * reliability_score
    ).round(2)

    supplier_stats["Status"] = np.select(
        [supplier_stats["Quality_Health_Score"] > 80, supplier_stats["Quality_Health_Score"].between(60, 80)],
        ["🟢 Strong", "🟡 Watch"],
        default="🔴 Critical",
    )

    scorecard = supplier_stats.reset_index()[
        [
            "Supplier_ID",
            "Defect_Rate",
            "Volume_CV",
            "Regional_Reliability",
            "Quality_Health_Score",
            "Status",
        ]
    ].sort_values("Quality_Health_Score", ascending=False)
    return scorecard


st.markdown('<div class="glow-title">⚙️ Manufacturing Defect & Quality Control Analytics</div>', unsafe_allow_html=True)
st.caption("Single-file edition with enhanced visuals and larger synthetic data.")

with st.sidebar:
    st.header("Control Panel")
    row_count = st.slider("Synthetic rows", min_value=1000, max_value=20000, value=5000, step=500)
    selected_regions = st.multiselect(
        "Filter regions", ["North", "South", "East", "West"], default=["North", "South", "East", "West"]
    )
    seed_value = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.subheader("💰 Financial Inputs")
    unit_manufacturing_cost = st.number_input("Unit Manufacturing Cost ($)", min_value=1.0, value=125.0, step=1.0)
    cost_per_defect_penalty = st.number_input("Cost per Defect Penalty ($)", min_value=1.0, value=420.0, step=5.0)
    animated_charts = st.toggle("Enable moving animated charts", value=True)

raw_df = generate_synthetic_data(rows=row_count, seed=int(seed_value))
processed_df = clean_transform_data(raw_df)
processed_df = processed_df[processed_df["Supplier_Region"].isin(selected_regions)]
relevant_df, unwanted_df = split_relevant_unwanted_data(processed_df)
analytics_df = relevant_df if len(relevant_df) > 0 else processed_df

supplier_defect = (
    analytics_df.groupby("Supplier_ID", as_index=False)["Defect_Rate"].mean().sort_values("Defect_Rate", ascending=False)
)
top5 = supplier_defect.head(5)

trend = analytics_df.groupby("Month", as_index=False)["Defect_Rate"].mean().sort_values("Month")

plant_fail = (
    analytics_df.assign(Is_Fail=analytics_df["Inspection_Result"].eq("Fail"))
    .groupby("Plant_Location", as_index=False)["Is_Fail"]
    .sum()
    .rename(columns={"Is_Fail": "Failure_Count"})
    .sort_values("Failure_Count", ascending=False)
)


monthly_supplier_defect = (
    analytics_df.groupby(["Month", "Supplier_ID"], as_index=False)["Defect_Rate"]
    .mean()
    .sort_values("Month")
)
monthly_top_suppliers = monthly_supplier_defect.merge(top5[["Supplier_ID"]], on="Supplier_ID", how="inner")

monthly_plant_fail = (
    analytics_df.assign(Is_Fail=analytics_df["Inspection_Result"].eq("Fail"))
    .groupby(["Month", "Plant_Location"], as_index=False)["Is_Fail"]
    .sum()
    .rename(columns={"Is_Fail": "Failure_Count"})
    .sort_values("Month")
)

f_stat, p_value = hypothesis_test_by_region(analytics_df)

main_tab, finance_tab, ml_tab, scorecard_tab, heatmap_tab = st.tabs(
    [
        "📊 Quality Dashboard",
        "💰 Financial Impact & ROI Simulation",
        "🔮 ML Risk Prediction",
        "🏆 Global Supplier Scorecard",
        "📍 Regional Quality Heatmap",
    ]
)

with main_tab:
    with st.expander("Preview Data"):
        st.dataframe(analytics_df.head(20), use_container_width=True)

    st.markdown(
        f"""
        <span class="status-chip status-green pulse-soft">✅ Relevant rows kept: {len(relevant_df):,}</span>
        <span class="status-chip status-red pulse-soft">🧹 Unwanted rows removed: {len(unwanted_df):,}</span>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Data Quality Triage (Green = Relevant, Red = Unwanted)"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🟢 Relevant Data")
            st.dataframe(relevant_df.head(20), use_container_width=True)
        with col_b:
            st.markdown("#### 🔴 Unwanted Data")
            st.dataframe(unwanted_df.head(20), use_container_width=True)

    st.subheader("KPI Overview")
    k1, k2, k3, k4 = st.columns(4)
    avg_defect = analytics_df["Defect_Rate"].mean()
    failures = (analytics_df["Inspection_Result"] == "Fail").sum()
    fail_ratio = failures / len(analytics_df) if len(analytics_df) else 0

    k1.metric("Rows", f"{len(analytics_df):,}")
    k2.metric("Avg Defect Rate", f"{avg_defect:.2%}")
    k3.metric("Fail Inspections", f"{failures:,}", delta=f"{fail_ratio:.2%} fail ratio")
    k4.metric("Suppliers", f"{analytics_df['Supplier_ID'].nunique()}")

    fig_top5 = px.bar(
        top5,
        x="Supplier_ID",
        y="Defect_Rate",
        text=top5["Defect_Rate"].map(lambda v: f"{v:.2%}"),
        color="Defect_Rate",
        color_continuous_scale="Turbo",
        title="🔥 Top 5 Suppliers with Highest Defect Rates",
    )
    fig_top5.update_traces(textposition="outside")
    fig_top5.update_layout(yaxis_tickformat=".1%", transition_duration=900)

    fig_trend = px.area(trend, x="Month", y="Defect_Rate", title="📈 Defect Trend Over Time", line_shape="spline")
    fig_trend.update_layout(yaxis_tickformat=".1%", transition_duration=900)

    fig_plant = px.pie(
        plant_fail,
        names="Plant_Location",
        values="Failure_Count",
        title="🏭 Quality Failure Distribution by Plant Location",
        hole=0.45,
    )
    fig_plant.update_layout(transition_duration=900)

    left, right = st.columns(2)
    left.plotly_chart(fig_top5, use_container_width=True)
    right.plotly_chart(fig_trend, use_container_width=True)
    st.plotly_chart(fig_plant, use_container_width=True)

    if animated_charts:
        st.subheader("✨ Moving Animated Charts")
        if len(monthly_top_suppliers) > 0 and len(monthly_plant_fail) > 0:
            anim_left, anim_right = st.columns(2)

            animated_top5 = px.bar(
                monthly_top_suppliers,
                x="Supplier_ID",
                y="Defect_Rate",
                color="Supplier_ID",
                animation_frame="Month",
                title="🎞️ Monthly Movement: Top Supplier Defect Rates",
                range_y=[0, max(0.05, float(monthly_top_suppliers["Defect_Rate"].max() * 1.15))],
            )
            animated_top5.update_layout(yaxis_tickformat=".1%")

            animated_plant = px.bar(
                monthly_plant_fail,
                x="Plant_Location",
                y="Failure_Count",
                color="Plant_Location",
                animation_frame="Month",
                title="🎞️ Monthly Movement: Plant Failure Distribution",
            )

            anim_left.plotly_chart(animated_top5, use_container_width=True)
            anim_right.plotly_chart(animated_plant, use_container_width=True)
        else:
            st.info("Not enough data after filtering to render moving animated charts.")

    st.subheader("Statistical Analysis")
    if np.isnan(p_value):
        st.warning("Not enough data to perform ANOVA.")
    else:
        gauge_value = min(max(avg_defect * 100, 0), 10)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": "Avg Defect Rate (%)"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#7CFC00" if avg_defect < 0.02 else "#FF7F50"},
                    "steps": [
                        {"range": [0, 2], "color": "#1f9d55"},
                        {"range": [2, 4], "color": "#facc15"},
                        {"range": [4, 10], "color": "#ef4444"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
        st.write(f"ANOVA F-statistic: **{f_stat:.4f}** | p-value: **{p_value:.8f}**")
        if p_value < 0.05:
            st.success("Defect rates differ significantly across regions.")
        else:
            st.success("No significant regional defect-rate differences detected.")

    st.subheader("Business Logic")
    worst_suppliers = ", ".join(top5["Supplier_ID"].tolist())

    st.markdown(
        f"""
<div class="insight-card"><b>1) Highest-risk suppliers:</b> {worst_suppliers}.</div>
<div class="insight-card"><b>2) Regional significance:</b> {'Significant' if p_value < 0.05 else 'Not significant'} (p = {p_value:.8f}).</div>
<div class="insight-card"><b>3) Strategy improvements:</b>
<ul>
<li>Deploy targeted supplier CAPA plans for the top-defect suppliers.</li>
<li>Apply tighter incoming inspection thresholds for suppliers with repeated failures.</li>
<li>Run region-specific quality audits and process capability checks.</li>
<li>Use performance-linked contracts tied to defect-rate reduction milestones.</li>
<li>Replicate best-practice SOPs from low-defect suppliers across the network.</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

with finance_tab:
    st.subheader("💰 Financial Impact & ROI Simulation")
    defect_reduction_pct = st.slider("Defect Reduction Target (%)", min_value=0, max_value=80, value=20, step=5)

    financial_summary = calculate_financial_impact(
        analytics_df,
        unit_cost=unit_manufacturing_cost,
        defect_penalty=cost_per_defect_penalty,
        reduction_pct=defect_reduction_pct,
    )
    plant_financials = financial_summary["plant_financials"]

    m1, m2 = st.columns(2)
    m1.metric("Total Quality Loss", f"${financial_summary['total_quality_loss']:,.0f}")
    m2.metric(
        f"Potential Savings ({defect_reduction_pct}% reduction)",
        f"${financial_summary['potential_savings']:,.0f}",
    )

    selected_plant = st.selectbox("Plant Location", plant_financials["Plant_Location"].tolist())
    selected_financial = plant_financials.loc[plant_financials["Plant_Location"] == selected_plant].iloc[0]

    waterfall = go.Figure(
        go.Waterfall(
            name=selected_plant,
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Gross Profit", "Quality Costs", "Net Adjusted Profit"],
            y=[
                selected_financial["Gross_Profit"],
                -selected_financial["Quality_Cost"],
                0,
            ],
            connector={"line": {"color": "rgba(255,255,255,0.4)"}},
        )
    )
    waterfall.update_layout(
        title=f"Gross Profit to Net Adjusted Profit - {selected_plant}",
        yaxis_title="USD ($)",
    )
    st.plotly_chart(waterfall, use_container_width=True)
    st.dataframe(plant_financials, use_container_width=True)

with ml_tab:
    st.subheader("🔮 ML Risk Prediction")
    model, model_columns = train_risk_model(analytics_df)

    supplier_region_map = (
        analytics_df.groupby("Supplier_ID")["Supplier_Region"].agg(lambda x: x.mode().iloc[0]).to_dict()
    )

    ml_left, ml_right = st.columns(2)
    with ml_left:
        selected_supplier = st.selectbox("Supplier", sorted(analytics_df["Supplier_ID"].unique()))
        default_volume = int(analytics_df["Production_Volume"].median())
        volume_options = sorted(analytics_df["Production_Volume"].round(-2).astype(int).unique())
        selected_volume = st.selectbox("Production Volume", volume_options, index=min(10, len(volume_options) - 1))

    inferred_region = supplier_region_map[selected_supplier]
    live_features = pd.DataFrame(
        {
            "Supplier_ID": [selected_supplier],
            "Supplier_Region": [inferred_region],
            "Production_Volume": [selected_volume if selected_volume else default_volume],
        }
    )
    if SKLEARN_AVAILABLE and model is not None:
        live_encoded = pd.get_dummies(live_features, drop_first=False).reindex(columns=model_columns, fill_value=0)
        fail_probability = float(model.predict_proba(live_encoded)[0][1])
        risk_score = fail_probability * 100
    else:
        risk_score = float(np.clip(analytics_df["Defect_Rate"].mean() * 3000, 0, 100))

    if risk_score >= 70:
        recommendation = "High Risk: Mandate 100% Manual Inspection"
    elif risk_score >= 40:
        recommendation = "Moderate Risk: Increase sampling and tighten in-process checks"
    else:
        recommendation = "Low Risk: Maintain standard QA cadence"

    with ml_right:
        st.metric("Risk Score", f"{risk_score:.1f}%")
        st.write(f"Inferred Supplier Region: **{inferred_region}**")
        st.info(f"Recommendation: **{recommendation}**")
        if not SKLEARN_AVAILABLE:
            st.warning(
                "scikit-learn is unavailable in this runtime; showing a defect-rate-based fallback risk score."
            )

with scorecard_tab:
    st.subheader("🏆 Global Supplier Scorecard")
    scorecard = build_supplier_scorecard(analytics_df)

    def highlight_quality_rows(row):
        score = row["Quality_Health_Score"]
        if score > 80:
            color = "background-color: rgba(34,197,94,0.35)"
        elif score >= 60:
            color = "background-color: rgba(250,204,21,0.35)"
        else:
            color = "background-color: rgba(239,68,68,0.35)"
        return [color] * len(row)

    styled_scorecard = (
        scorecard.style.format(
            {
                "Defect_Rate": "{:.2%}",
                "Volume_CV": "{:.2f}",
                "Regional_Reliability": "{:.2%}",
                "Quality_Health_Score": "{:.2f}",
            }
        )
        .apply(highlight_quality_rows, axis=1)
        .hide(axis="index")
    )

    st.dataframe(styled_scorecard, use_container_width=True)

with heatmap_tab:
    st.subheader("📍 Regional Quality Heatmap")
    region_coordinates = {
        "North": {"lat": 45.0, "lon": -93.0},
        "South": {"lat": 32.5, "lon": -95.0},
        "East": {"lat": 40.0, "lon": -75.0},
        "West": {"lat": 37.5, "lon": -120.0},
    }

    regional_view = (
        analytics_df.groupby("Supplier_Region", as_index=False)
        .agg(Production_Volume=("Production_Volume", "sum"), Defect_Rate=("Defect_Rate", "mean"))
        .rename(columns={"Supplier_Region": "Region"})
    )

    top_failing_plant = (
        analytics_df.assign(Fail_Flag=analytics_df["Inspection_Result"].eq("Fail").astype(int))
        .groupby(["Supplier_Region", "Plant_Location"], as_index=False)["Fail_Flag"]
        .sum()
        .sort_values(["Supplier_Region", "Fail_Flag"], ascending=[True, False])
        .drop_duplicates("Supplier_Region")
        .rename(columns={"Supplier_Region": "Region", "Plant_Location": "Top_Failing_Plant"})
    )

    regional_view["Latitude"] = regional_view["Region"].map(lambda x: region_coordinates[x]["lat"])
    regional_view["Longitude"] = regional_view["Region"].map(lambda x: region_coordinates[x]["lon"])
    regional_view = regional_view.merge(top_failing_plant[["Region", "Top_Failing_Plant"]], on="Region", how="left")

    geo = px.scatter_geo(
        regional_view,
        lat="Latitude",
        lon="Longitude",
        size="Production_Volume",
        color="Defect_Rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="Region",
        hover_data={
            "Production_Volume": ":,.0f",
            "Defect_Rate": ":.2%",
            "Top_Failing_Plant": True,
            "Latitude": False,
            "Longitude": False,
        },
        title="Regional Quality Heatmap (Bubble size = Production Volume, Color = Defect Rate)",
        projection="natural earth",
    )
    st.plotly_chart(geo, use_container_width=True)