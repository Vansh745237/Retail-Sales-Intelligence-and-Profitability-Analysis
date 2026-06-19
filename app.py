import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import os

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

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
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.3);
}

.stTabs [data-baseweb="tab"]{
background:#1e293b;
color:white;
border-radius:10px;
margin-right:5px;
}

.stTabs [aria-selected="true"]{
background:#2563eb;
color:white;
}

h1,h2,h3{
color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD DATA
# ---------------------------------

df = pd.read_csv("data/SampleSuperstore.csv")

# ---------------------------------
# HELPER
# ---------------------------------

def dark_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font_color="white"
    )
    return fig

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = None

if os.path.exists("models/profit_prediction_model.pkl"):
    model = joblib.load("models/profit_prediction_model.pkl")

# ---------------------------------
# HERO SECTION
# ---------------------------------

st.markdown("""
<div style="
padding:35px;
border-radius:20px;
background:linear-gradient(135deg,#2563eb,#06b6d4);
text-align:center;
margin-bottom:20px;
box-shadow:0px 6px 20px rgba(0,0,0,0.4);">

<h1 style="color:white;">
📊 Retail Sales Intelligence Dashboard
</h1>

<p style="font-size:18px;color:white;">
Business Intelligence • Analytics • Profit Prediction
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.markdown("""
# 📈 Analytics Hub

Retail Intelligence Dashboard
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
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

# ---------------------------------
# KPI SECTION
# ---------------------------------

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_quantity = filtered_df["Quantity"].sum()

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
col3.metric("📦 Quantity Sold", f"{total_quantity:,.0f}")
col4.metric("🎯 Profit Margin", f"{profit_margin:.2f}%")

st.divider()

# ---------------------------------
# TABS
# ---------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Sales Analytics",
        "💰 Profitability",
        "🧠 Prediction"
    ]
)

# ---------------------------------
# OVERVIEW
# ---------------------------------

with tab1:

    c1, c2 = st.columns(2)

    with c1:

        sales_region = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            sales_region,
            x="Region",
            y="Sales",
            title="Sales by Region"
        )

        fig = dark_plot(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:

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
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# SALES ANALYTICS
# ---------------------------------

with tab2:

    sales_category = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        sales_category,
        names="Category",
        values="Sales",
        title="Sales Distribution by Category"
    )

    fig = dark_plot(fig)
    st.plotly_chart(fig, use_container_width=True)

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
        title="Top 10 States by Sales"
    )

    fig = dark_plot(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# PROFITABILITY
# ---------------------------------

with tab3:

    fig = px.scatter(
        filtered_df,
        x="Discount",
        y="Profit",
        title="Discount vs Profit"
    )

    fig = dark_plot(fig)
    st.plotly_chart(fig, use_container_width=True)

    corr_df = filtered_df[
        ["Sales", "Quantity", "Discount", "Profit"]
    ].corr()

    fig = ff.create_annotated_heatmap(
        z=corr_df.values,
        x=list(corr_df.columns),
        y=list(corr_df.index),
        annotation_text=round(corr_df, 2).values
    )

    fig.update_layout(
        template="plotly_dark",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# PREDICTION
# ---------------------------------

with tab4:

    st.subheader("🧠 Profit Prediction")

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

    if st.button("Predict Profit"):

        if model is not None:

            prediction = model.predict(
                [[sales, quantity, discount]]
            )[0]

            st.success(
                f"Expected Profit: ${prediction:.2f}"
            )

        else:

            st.warning(
                "profit_prediction_model.pkl not found."
            )

# ---------------------------------
# BUSINESS INSIGHTS
# ---------------------------------

st.divider()

st.subheader("📋 Business Insights")

st.markdown("""
✅ West region generated the highest sales

✅ Discounts negatively impacted profitability

✅ Top-performing states contributed a large share of revenue

✅ Category performance varied significantly

✅ Data-driven decision making can improve profitability
""")