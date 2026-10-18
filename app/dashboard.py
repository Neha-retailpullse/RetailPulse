

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.express as px
from src.preprocessing import load_data, daily_sales
from src.segmentation import build_rfm, segment_customers
from src.forecasting import train_forecast

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FC;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    h1 {
        color: #17365D !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #24527A !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE6F1;
        padding: 22px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(23, 54, 93, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #52657A !important;
        font-size: 14px !important;
    }

    [data-testid="stMetricValue"] {
        color: #17365D !important;
        font-weight: 750 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE6F1;
        border-radius: 12px;
        overflow: hidden;
        background-color: #FFFFFF;
    }

    [data-testid="stSelectbox"],
    [data-testid="stDateInput"] {
        background-color: #FFFFFF;
        border-radius: 10px;
    }

    [data-testid="stCaptionContainer"] {
        color: #64748B;
    }

    hr {
        border-color: #DCE6F1;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")


df = load_data()
st.write("Available columns:", df.columns.tolist())
# Interactive dashboard filters
st.subheader("🔎 Dashboard Filters")

import pandas as pd
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

col_date, col_product = st.columns(2)

with col_date:
    start_date, end_date = st.date_input(
        "Select date range",
       value=(df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()),
min_value=df["InvoiceDate"].min().date(),
max_value=df["InvoiceDate"].max().date(),
    )

with col_product:
    products = ["All Products"] + sorted(df["ProductName"].unique().tolist())
    selected_product = st.selectbox("Select product", products)


# Apply filters
df = df[
    (df["InvoiceDate"].dt.date >= start_date)
    & (df["InvoiceDate"].dt.date <= end_date)
]


  

if selected_product != "All Products":
    df = df[df["ProductName"] == selected_product]

if df.empty:
    st.warning("No sales found for the selected filters.")
    st.stop()

# KPIs
revenue = df["Revenue"].sum()
orders = len(df)
customers = df["CustomerID"].nunique()
units = df["Quantity"].sum()

# Display KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div style="background:#FFFFFF; padding:22px; border-radius:12px;
                    border:1px solid #E2E8F0; border-top:4px solid #2563EB;">
            <p style="color:#64748B; margin:0; font-size:14px;">Total Revenue</p>
            <h2 style="color:#17365D; margin:8px 0 0 0;">₹{revenue:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background:#FFFFFF; padding:22px; border-radius:12px;
                    border:1px solid #E2E8F0; border-top:4px solid #0F766E;">
            <p style="color:#64748B; margin:0; font-size:14px;">Transactions</p>
            <h2 style="color:#17365D; margin:8px 0 0 0;">{orders:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="background:#FFFFFF; padding:22px; border-radius:12px;
                    border:1px solid #E2E8F0; border-top:4px solid #7C3AED;">
            <p style="color:#64748B; margin:0; font-size:14px;">Customers</p>
            <h2 style="color:#17365D; margin:8px 0 0 0;">{customers:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div style="background:#FFFFFF; padding:22px; border-radius:12px;
                    border:1px solid #E2E8F0; border-top:4px solid #D97706;">
            <p style="color:#64748B; margin:0; font-size:14px;">Units Sold</p>
            <h2 style="color:#17365D; margin:8px 0 0 0;">{units:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# Inventory Alerts
st.subheader("📦 Inventory Alerts")

inventory = (
    df.groupby(["ProductID", "ProductName"], as_index=False)
      .agg(
          CurrentStock=("StockQuantity", "last"),
          UnitsSold=("Quantity", "sum")
      )
)

inventory["Status"] = inventory.apply(
    lambda r: "🔴 Reorder"
    if r["CurrentStock"] < max(20, r["UnitsSold"] / 20)
    else (
        "🟡 Monitor"
        if r["CurrentStock"] < max(40, r["UnitsSold"] / 10)
        else "🟢 Healthy"
    ),
    axis=1
)

# Show inventory summary
reorder_count = (inventory["Status"] == "🔴 Reorder").sum()
monitor_count = (inventory["Status"] == "🟡 Monitor").sum()
healthy_count = (inventory["Status"] == "🟢 Healthy").sum()

col1, col2, col3 = st.columns(3)

col1.metric("🔴 Reorder Needed", reorder_count)
col2.metric("🟡 Needs Monitoring", monitor_count)
col3.metric("🟢 Healthy Stock", healthy_count)

st.divider()

# Display products needing attention first
status_order = {
    "🔴 Reorder": 0,
    "🟡 Monitor": 1,
    "🟢 Healthy": 2
}

inventory["Priority"] = inventory["Status"].map(status_order)

inventory = inventory.sort_values(
    ["Priority", "CurrentStock"],
    ascending=[True, True]
)


inventory_display = inventory.drop(columns=["Priority"]).rename(
    columns={
        "ProductID": "Product ID",
        "ProductName": "Product Name",
        "CurrentStock": "Current Stock",
        "UnitsSold": "Units Sold",
        "Status": "Stock Status"
    }
)

# Stock Status Filter
st.markdown("### 🔎 Filter Inventory")

status_options = [
    "All",
    "🔴 Reorder",
    "🟡 Monitor",
    "🟢 Healthy"
]

selected_status = st.selectbox(
    "Select stock status",
    status_options
)

# Filter products based on selected status
if selected_status == "All":
    filtered_inventory = inventory_display
else:
    filtered_inventory = inventory_display[
        inventory_display["Stock Status"] == selected_status
    ]

# Show filtered inventory
st.dataframe(
    filtered_inventory,
    use_container_width=True,
    hide_index=True
)

st.caption(f"Showing {len(filtered_inventory)} products")


st.divider()


# Sales Insights
st.subheader("💡 Sales Insights")

# Calculate average revenue per transaction
average_order_value = revenue / orders if orders > 0 else 0

# Find the best-selling product
product_sales = (
    df.groupby("ProductName", as_index=False)
      .agg(TotalUnits=("Quantity", "sum"))
      .sort_values("TotalUnits", ascending=False)
)

if not product_sales.empty:
    best_product = product_sales.iloc[0]["ProductName"]
    best_product_units = product_sales.iloc[0]["TotalUnits"]
else:
    best_product = "No sales data"
    best_product_units = 0

# Display insight cards
insight1, insight2 = st.columns(2)

with insight1:
    st.metric("Average Order Value", f"₹{average_order_value:,.2f}")

with insight2:
    st.metric("Best-Selling Product", str(best_product))

st.info(
    f"Your store recorded ₹{revenue:,.2f} in revenue across "
    f"{orders:,} transactions. The best-selling product is "
    f"{best_product}, with {best_product_units:,.0f} units sold."
)

st.divider()

# Sales Trend
st.subheader("📈 Sales Trend")
daily = daily_sales(df)
fig = px.line(
    daily,
    x="ds",
    y="y",
    labels={"ds": "Date", "y": "Revenue"},
    template="plotly_white",
    title="Daily Revenue"
)

fig.update_traces(
    line=dict(color="#2563EB", width=3),
    hovertemplate="Date: %{x|%d %b %Y}<br>Revenue: %{y:,.2f}<extra></extra>"
)

fig.update_layout(
    title_font=dict(size=20, color="#17365D"),
    font=dict(family="Arial", color="#52657A"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=20, r=20, t=65, b=20),
    xaxis=dict(showgrid=False, title="Date"),
    yaxis=dict(showgrid=True, gridcolor="#E8EEF5", title="Revenue")
)

st.plotly_chart(fig, use_container_width=True)

# Demand Forecast
# Demand Forecast
st.subheader("🔮 30-Day Demand Forecast")

try:
    model, forecast = train_forecast(daily, periods=30)
    forecast_view = forecast.tail(60)

    fig2 = px.line(
        forecast_view,
        x="ds",
        y="yhat",
        labels={"ds": "Date", "yhat": "Forecast Revenue"},
        title="Historical + Forecasted Revenue",
        template="plotly_white"
    )

    fig2.update_traces(
        line=dict(color="#0F766E", width=3)
    )

    fig2.update_layout(
        title_font=dict(size=20, color="#17365D"),
        font=dict(family="Arial", color="#52657A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#E8EEF5", title="Forecast Revenue")
    )

    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"Forecasting could not run yet: {e}")


# Customer Segmentation
st.subheader("👥 Customer Segmentation")

# Build customer segments
rfm = build_rfm(df)
segmented, _ = segment_customers(rfm)

# Summarize each customer cluster
cluster_summary = (
    segmented.groupby("Cluster")
    .agg(
        Customers=("CustomerID", "nunique"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean")
    )
    .reset_index()
)

# Add business-friendly labels to customer groups
high_value_cluster = (
    cluster_summary["AvgMonetary"] + cluster_summary["AvgFrequency"]
).idxmax()

needs_attention_cluster = cluster_summary["AvgRecency"].idxmax()

cluster_summary["Customer Group"] = "Regular Customers"
cluster_summary["Recommendation"] = "Monitor purchases and encourage repeat orders."

# Identify high-value customers
cluster_summary.loc[
    cluster_summary.index == high_value_cluster,
    ["Customer Group", "Recommendation"]
] = [
    "High-Value Customers",
    "Reward with loyalty benefits and personalized offers."
]

# Identify customers who may need re-engagement
if needs_attention_cluster != high_value_cluster:
    cluster_summary.loc[
        cluster_summary.index == needs_attention_cluster,
        ["Customer Group", "Recommendation"]
    ] = [
        "Needs Re-engagement",
        "Send a win-back offer to encourage another purchase."
    ]

# Identify frequent buyers who are not already in another group
remaining = cluster_summary[
    ~cluster_summary.index.isin(
        [high_value_cluster, needs_attention_cluster]
    )
]

if not remaining.empty:
    loyal_cluster = remaining["AvgFrequency"].idxmax()

    cluster_summary.loc[
        cluster_summary.index == loyal_cluster,
        ["Customer Group", "Recommendation"]
    ] = [
        "Frequent Buyers",
        "Use loyalty rewards to encourage continued purchases."
    ]

# Round average values for readability
cluster_summary["AvgRecency"] = cluster_summary["AvgRecency"].round(1)
cluster_summary["AvgFrequency"] = cluster_summary["AvgFrequency"].round(1)
cluster_summary["AvgMonetary"] = cluster_summary["AvgMonetary"].round(2)

# Show customer overview
total_customers = segmented["CustomerID"].nunique()
total_clusters = segmented["Cluster"].nunique()

col1, col2 = st.columns(2)

with col1:
    st.metric("Customers Segmented", f"{total_customers:,}")

with col2:
    st.metric("Customer Groups", f"{total_clusters:,}")

st.divider()

# Customer frequency vs spending chart
st.markdown("### 📊 Customer Spending Patterns")

fig3 = px.scatter(
    segmented,
    x="Frequency",
    y="Monetary",
    color=segmented["Cluster"].astype(str),
    hover_data=["CustomerID", "Recency"],
    labels={
        "Frequency": "Purchase Frequency",
        "Monetary": "Total Spending",
        "color": "Customer Group",
        "Recency": "Days Since Last Purchase"
    },
    title="Customer Groups by Purchase Frequency and Spending",
    template="plotly_white"
)

fig3.update_layout(
    title_font=dict(size=18, color="#17365D"),
    font=dict(family="Arial", color="#52657A"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=20, r=20, t=65, b=20),
    xaxis=dict(showgrid=True, gridcolor="#E8EEF5"),
    yaxis=dict(showgrid=True, gridcolor="#E8EEF5")
)

st.plotly_chart(fig3, use_container_width=True)

# Customer group summary table
st.markdown("### 📋 Customer Group Summary")

cluster_display = cluster_summary.rename(
    columns={
        "Cluster": "Group ID",
        "Customers": "Number of Customers",
        "AvgRecency": "Avg. Days Since Purchase",
        "AvgFrequency": "Avg. Purchase Frequency",
        "AvgMonetary": "Avg. Customer Spending"
    }
)

st.dataframe(
    cluster_display,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "RetailPulse MVP • Customer groups are generated using RFM features "
    "and the existing clustering model."
)

# ---------------- PRODUCT ANALYTICS ----------------
st.divider()
st.header("📦 Product Analytics")

product_summary = (
    df.groupby(["ProductID", "ProductName", "Category"], as_index=False)
    .agg(
        Units_Sold=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Current_Stock=("StockQuantity", "last")
    )
    .sort_values("Total_Revenue", ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Products by Revenue")
    fig_product = px.bar(
        product_summary.head(10),
        x="ProductName",
        y="Total_Revenue",
        color="Category",
        title="Top 10 Products by Revenue",
        labels={
            "ProductName": "Product",
            "Total_Revenue": "Revenue"
        },
        template="plotly_white"
    )
    st.plotly_chart(fig_product, use_container_width=True)

with col2:
    st.subheader("Revenue by Category")
    category_summary = (
        df.groupby("Category", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )
    fig_category = px.pie(
        category_summary,
        names="Category",
        values="Revenue",
        title="Revenue Share by Category",
        template="plotly_white"
    )
    st.plotly_chart(fig_category, use_container_width=True)

st.subheader("Product Performance")
st.dataframe(
    product_summary.rename(columns={
        "ProductID": "Product ID",
        "ProductName": "Product Name",
        "Units_Sold": "Units Sold",
        "Total_Revenue": "Total Revenue",
        "Current_Stock": "Stock Quantity"
    }),
    use_container_width=True,
    hide_index=True
)

# ---------------- INVENTORY ALERTS ----------------
st.divider()
st.header("🚨 Inventory Alerts")

stock_threshold = st.slider(
    "Low-stock alert threshold",
    min_value=10,
    max_value=100,
    value=30,
    step=5
)

inventory = product_summary.copy()

inventory["Stock Status"] = inventory["Current_Stock"].apply(
    lambda stock: "🔴 Low Stock"
    if stock <= stock_threshold
    else "🟢 In Stock"
)

low_stock = inventory[
    inventory["Current_Stock"] <= stock_threshold
].sort_values("Current_Stock")

col1, col2 = st.columns(2)

with col1:
    st.metric("Products Tracked", len(inventory))

with col2:
    st.metric("Low-Stock Products", len(low_stock))

if low_stock.empty:
    st.success("No products are below the selected stock threshold.")
else:
    st.warning(
        f"{len(low_stock)} products have stock at or below "
        f"{stock_threshold} units."
    )
    st.dataframe(
        low_stock[[
            "ProductID",
            "ProductName",
            "Category",
            "Current_Stock",
            "Units_Sold",
            "Stock Status"
        ]].rename(columns={
            "ProductID": "Product ID",
            "ProductName": "Product Name",
            "Current_Stock": "Stock Quantity",
            "Units_Sold": "Units Sold"
        }),
        use_container_width=True,
        hide_index=True
    )


