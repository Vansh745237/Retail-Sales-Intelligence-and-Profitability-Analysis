import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import os

st.set_page_config(
    page_title="Retail Sales Intelligence Suite",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp{background:#020617;color:white;}
.block-container{padding-top:0.7rem;max-width:95%;}
header{visibility:hidden;}
section[data-testid="stSidebar"]{
    background:#0f172a;
    border-right:1px solid #1e293b;
}
span[data-baseweb="tag"]{
    background:#2563eb !important;
    color:white !important;
}
div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #334155;
    border-radius:16px;
    padding:18px;
}
div[data-testid="stMetricValue"]{
    color:white !important;
    font-size:30px !important;
    font-weight:700 !important;
}
h1,h2,h3,h4,h5,h6,p{color:white !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/SampleSuperstore.csv")

df = load_data()

model = None
if os.path.exists("models/profit_prediction_model.pkl"):
    try:
        model = joblib.load("models/profit_prediction_model.pkl")
    except:
        model = None

def dark_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font_color="white"
    )
    return fig

st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a,#2563eb);
padding:22px;border-radius:18px;margin-bottom:20px;">
<h1>🚀 Retail Sales Intelligence Suite</h1>
<p>Executive Analytics • Business Intelligence • AI Forecasting</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🚀 Analytics Control Center")

region = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), default=df["Category"].unique())
segment = st.sidebar.multiselect("Segment", df["Segment"].unique(), default=df["Segment"].unique())

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_quantity = filtered_df["Quantity"].sum()
orders = len(filtered_df)

profit_margin = (total_profit/total_sales*100) if total_sales else 0
avg_order = (total_sales/orders) if orders else 0

best_region = "N/A"
if len(filtered_df):
    best_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()

r1c1,r1c2,r1c3 = st.columns(3)
r2c1,r2c2,r2c3 = st.columns(3)

r1c1.metric("Revenue", f"${total_sales:,.0f}")
r1c2.metric("Profit", f"${total_profit:,.0f}")
r1c3.metric("Orders", f"{orders:,}")
r2c1.metric("Quantity", f"{total_quantity:,}")
r2c2.metric("Margin", f"{profit_margin:.2f}%")
r2c3.metric("Top Region", best_region)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Executive Dashboard","📈 Sales Analytics","💰 Profitability","🤖 AI Forecasting"]
)

with tab1:
    c1,c2 = st.columns(2)

    sales_region = filtered_df.groupby("Region")["Sales"].sum().reset_index()
    profit_region = filtered_df.groupby("Region")["Profit"].sum().reset_index()

    with c1:
        fig = px.bar(sales_region,x="Region",y="Sales",title="Revenue by Region",text_auto=".2s")
        st.plotly_chart(dark_plot(fig), use_container_width=True)

    with c2:
        fig = px.bar(profit_region,x="Region",y="Profit",title="Profit by Region",text_auto=".2s")
        st.plotly_chart(dark_plot(fig), use_container_width=True)

with tab2:
    c1,c2 = st.columns(2)

    sales_category = filtered_df.groupby(["Category","Sub-Category"])["Sales"].sum().reset_index()

    with c1:
        fig = px.treemap(
            sales_category,
            path=["Category","Sub-Category"],
            values="Sales",
            title="Category Revenue Contribution"
        )
        st.plotly_chart(dark_plot(fig), use_container_width=True)

    with c2:
        top_states = filtered_df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_states,x="State",y="Sales",title="Top States by Revenue")
        st.plotly_chart(dark_plot(fig), use_container_width=True)

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
    st.plotly_chart(dark_plot(fig), use_container_width=True)

    corr = filtered_df[["Sales","Quantity","Discount","Profit"]].corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=round(corr,2).values
    )
    fig.update_layout(template="plotly_dark", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🤖 AI Profit Forecast Engine")

    sales = st.number_input("Sales", min_value=0.0, value=500.0)
    quantity = st.number_input("Quantity", min_value=1, value=5)
    discount = st.slider("Discount", 0.0, 1.0, 0.10)

    if st.button("Generate Forecast"):
        if model is not None:
            pred = model.predict([[sales, quantity, discount]])[0]
            st.metric("Predicted Profit", f"${pred:.2f}")

            if pred > 500:
                st.success("High Profit Opportunity")
            elif pred > 0:
                st.info("Moderate Profit Opportunity")
            else:
                st.error("Potential Loss Risk")
        else:
            st.warning("Model file not available.")

st.divider()
st.subheader("📋 Executive Insights")

if len(filtered_df):
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    top_state = filtered_df.groupby("State")["Sales"].sum().idxmax()

    st.success(f"Highest revenue generated by {top_region} region.")
    st.info(f"Top performing category: {top_category}.")
    st.info(f"Best performing state: {top_state}.")

st.markdown("---")
st.markdown("### 👨‍💻 Developed By Vansh Bathla")
