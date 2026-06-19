import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Retail Sales Intelligence Suite",
    page_icon="📊",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:#0b1120;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

div[data-testid="metric-container"]{
    background:linear-gradient(135deg,#1e293b,#334155);
    border:1px solid #475569;
    padding:15px;
    border-radius:16px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.3);
}

h1,h2,h3,h4,h5,h6{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("data/SampleSuperstore.csv")

df = load_data()

# =====================================
# LOAD MODEL
# =====================================

model = None

if os.path.exists("models/profit_prediction_model.pkl"):
    model = joblib.load(
        "models/profit_prediction_model.pkl"
    )

# =====================================
# HELPER
# =====================================

def dark_plot(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font_color="white"
    )

    return fig

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#0f172a,#1e3a8a);
padding:35px;
border-radius:22px;
margin-bottom:20px;
">

<h1>
🚀 Retail Sales Intelligence Suite
</h1>

<p style="font-size:18px;color:#cbd5e1;">
Executive Analytics • Business Intelligence • AI Forecasting
</p>

</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.markdown("""
# 🚀 Analytics Control Center

Filter business metrics and explore insights.
""")

region = st.sidebar.multiselect(
    "Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

segment = st.sidebar.multiselect(
    "Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique()
)

filtered_df = df[
    (df["Region"].isin(region))
    &
    (df["Category"].isin(category))
    &
    (df["Segment"].isin(segment))
]

# =====================================
# KPI METRICS
# =====================================

total_sales = filtered_df["Sales"].sum()

total_profit = filtered_df["Profit"].sum()

total_quantity = filtered_df["Quantity"].sum()

orders = len(filtered_df)

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0
    else 0
)

avg_order_value = (
    total_sales / orders
    if orders != 0
    else 0
)

best_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
    if len(filtered_df) > 0
    else "N/A"
)

c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric(
    "💰 Revenue",
    f"${total_sales:,.0f}"
)

c2.metric(
    "📈 Profit",
    f"${total_profit:,.0f}"
)

c3.metric(
    "📦 Orders",
    f"{orders:,}"
)

c4.metric(
    "🎯 Margin",
    f"{profit_margin:.2f}%"
)

c5.metric(
    "🛒 Avg Order",
    f"${avg_order_value:,.0f}"
)

c6.metric(
    "🌍 Top Region",
    best_region
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Executive Dashboard",
        "📈 Sales Analytics",
        "💰 Profitability",
        "🤖 AI Forecasting"
    ]
)
# =====================================
# EXECUTIVE DASHBOARD
# =====================================

with tab1:

    left, right = st.columns(2)

    with left:

        sales_region = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            sales_region,
            x="Region",
            y="Sales",
            title="Revenue by Region"
        )

        fig = dark_plot(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        profit_region = (
            filtered_df.groupby("Region")["Profit"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            profit_region,
            x="Region",
            y="Profit",
            title="Profit by Region"
        )

        fig = dark_plot(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### 🎯 Executive Insights")

    top_region = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    top_category = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    st.success(
        f"Highest revenue generated by **{top_region} Region**."
    )

    st.info(
        f"Top performing category is **{top_category}**."
    )

# =====================================
# SALES ANALYTICS
# =====================================

with tab2:

    left, right = st.columns(2)

    with left:

        sales_category = (
            filtered_df.groupby(
                ["Category", "Sub-Category"]
            )["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.treemap(
            sales_category,
            path=["Category", "Sub-Category"],
            values="Sales",
            title="Category Revenue Contribution"
        )

        fig = dark_plot(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        top_states = (
            filtered_df.groupby("State")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_states,
            x="State",
            y="Sales",
            title="Top 10 States by Revenue"
        )

        fig = dark_plot(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### 📦 Category Performance")

    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        title="Sales by Category"
    )

    fig = dark_plot(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # =====================================
# PROFITABILITY ANALYSIS
# =====================================

with tab3:

    fig = px.scatter(
        filtered_df,
        x="Discount",
        y="Profit",
        size="Sales",
        color="Category",
        hover_data=["State"],
        title="Discount Impact on Profitability"
    )

    fig = dark_plot(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### 🔍 Correlation Analysis")

    corr_df = filtered_df[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Profit"
        ]
    ].corr()

    fig = ff.create_annotated_heatmap(
        z=corr_df.values,
        x=list(corr_df.columns),
        y=list(corr_df.index),
        annotation_text=round(corr_df, 2).values
    )

    fig.update_layout(
        template="plotly_dark",
        title="Business Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# AI FORECASTING
# =====================================

with tab4:

    st.subheader(
        "🤖 AI Profit Forecast Engine"
    )

    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=500.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=5
    )

    discount = st.slider(
        "Discount",
        0.0,
        1.0,
        0.10
    )

    st.info("""
💡 Enter Sales, Quantity and Discount values
to estimate expected profit using the trained ML model.
""")

    if st.button(
        "Generate Forecast"
    ):

        if model is not None:

            prediction = model.predict(
                [[
                    sales,
                    quantity,
                    discount
                ]]
            )[0]

            st.metric(
                "Predicted Profit",
                f"${prediction:.2f}"
            )

            if prediction > 500:

                st.success(
                    "High Profit Opportunity"
                )

            elif prediction > 0:

                st.info(
                    "Moderate Profit Opportunity"
                )

            else:

                st.error(
                    "Potential Loss Risk"
                )

        else:

            st.warning(
                "profit_prediction_model.pkl not found."
            )

# =====================================
# BUSINESS INSIGHTS
# =====================================

st.divider()

st.subheader(
    "📋 Executive Business Insights"
)
if len(filtered_df) > 0:
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    best_state = filtered_df.groupby("State")["Sales"].sum().idxmax()
else:
    top_region = top_category = best_state = "N/A"


st.success(
    f"Highest revenue generated by **{top_region} Region**."
)

st.info(
    f"Top performing category is **{top_category}**."
)

st.info(
    f"Best performing state is **{best_state}**."
)

if profit_margin > 15:

    st.success(
        "Profitability is healthy across selected segments."
    )

else:

    st.warning(
        "Profit margin indicates optimization opportunities."
    )

# =====================================
# FOOTER
# =====================================

st.markdown("""
---

### 👨‍💻 Developed By

**Vansh Bathla**

Data Science • Machine Learning • Business Intelligence

GitHub: Vansh745237

---
""")