import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import load_data, daily_sales, add_sale
from src.segmentation import build_rfm, segment_customers
from src.forecasting import train_forecast

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #F4F7FC; }
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
h1 { color: #17365D !important; font-weight: 800 !important; }
h2, h3 { color: #24527A !important; }
[data-testid="stMetric"] { background: white; border: 1px solid #DCE6F1; padding: 18px; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")
st.subheader("📌 Business Overview")

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

# PASTE THE ADD NEW SALE FORM CODE HERE

if df is None or df.empty:
    st.warning("The dataset is empty.")
    st.stop()
if df is None or df.empty:
    st.warning("The dataset is empty.")
    st.stop()
# Add New Sale
st.subheader("➕ Add New Sale")

with st.expander("Record a new sale"):
    products_df = load_data()

    with st.form("add_sale_form"):
        product_options = products_df[
            ["ProductID", "ProductName"]
        ].drop_duplicates()

        selected_product = st.selectbox(
            "Select Product",
            product_options["ProductID"].tolist(),
            format_func=lambda x: (
                f"{x} - "
                f"{product_options.loc[product_options['ProductID'] == x, 'ProductName'].iloc[0]}"
            )
        )

        customer_id = st.text_input("Customer ID")
        sale_date = st.date_input("Sale Date")
        quantity = st.number_input("Quantity", min_value=1, step=1)

        region = st.selectbox(
            "Region",
            sorted(products_df["Region"].dropna().unique())
        )

        submitted = st.form_submit_button("Save Sale")
        if submitted:
            try:
                add_sale(
                    sale_date,
                    customer_id,
                    selected_product,
                    int(quantity),
                    region
                )
                st.success("Sale recorded successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not record sale: {exc}")
required = ["InvoiceDate", "Revenue", "Quantity", "CustomerID"]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
df = df.dropna(subset=["InvoiceDate"])
if df.empty:
    st.warning("No valid invoice dates found.")
    st.stop()

# Filters
st.subheader("🔎 Dashboard Filters")
date_min, date_max = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
date_col, product_col = st.columns(2)
with date_col:
    selected_dates = st.date_input("Invoice date range", value=(date_min, date_max),
                                   min_value=date_min, max_value=date_max)
product_col_name = "ProductName" if "ProductName" in df.columns else (
    "ProductID" if "ProductID" in df.columns else None)
with product_col:
    product_options = sorted(df[product_col_name].dropna().astype(str).unique()) if product_col_name else []
    selected_products = st.multiselect("Products (empty = all)", product_options)

filtered = df.copy()
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    filtered = filtered[filtered["InvoiceDate"].dt.date.between(selected_dates[0], selected_dates[1])]
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    filtered = filtered[filtered["InvoiceDate"].dt.date == selected_dates[0]]
if product_col_name and selected_products:
    filtered = filtered[filtered[product_col_name].astype(str).isin(selected_products)]
if filtered.empty:
    st.info("No data matches these filters. Change the date range or product selection.")
    st.stop()

# KPIs
st.subheader("📌 Business Overview")
revenue = filtered["Revenue"].sum()
transactions = filtered["InvoiceNo"].nunique() if "InvoiceNo" in filtered.columns else len(filtered)
customers = filtered["CustomerID"].nunique()
units = filtered["Quantity"].sum()
avg_transaction = revenue / transactions if transactions else 0
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"₹{revenue:,.0f}")
k2.metric("Transactions", f"{transactions:,.0f}")
k3.metric("Customers", f"{customers:,.0f}")
k4.metric("Units Sold", f"{units:,.0f}")
k5.metric("Avg. Transaction Value", f"₹{avg_transaction:,.0f}")
st.divider()

# Sales insights and trend
st.subheader("💡 Sales Insights")
i1, i2, i3 = st.columns(3)
if "ProductName" in filtered.columns:
    product_revenue = filtered.groupby("ProductName")["Revenue"].sum().sort_values(ascending=False)
    top_name = str(product_revenue.index[0]) if not product_revenue.empty else "N/A"
    top_value = float(product_revenue.iloc[0]) if not product_revenue.empty else 0
else:
    top_name, top_value = "Not available", 0
daily = daily_sales(filtered)

daily["ds"] = pd.to_datetime(daily["ds"])
daily = (
    daily.sort_values("ds")
    .set_index("ds")
    .asfreq("D", fill_value=0)
    .rename_axis("ds")
    .reset_index()
)
best_day = daily.loc[daily["y"].idxmax()] if not daily.empty else None
i1.metric("Top Product by Revenue", top_name, help=f"Revenue: ₹{top_value:,.0f}")
i2.metric("Highest-Revenue Day", pd.to_datetime(best_day["ds"]).strftime("%d %b %Y") if best_day is not None else "N/A",
          f"₹{best_day['y']:,.0f}" if best_day is not None else None)
i3.metric("Unique Products", f"{filtered[product_col_name].nunique():,}" if product_col_name else "N/A")

st.subheader("📈 Sales Trend")
if not daily.empty:
    fig = px.line(daily, x="ds", y="y", markers=True, labels={"ds": "Date", "y": "Revenue"}, title="Daily Revenue")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Not enough sales data for a trend.")


# Forecast
st.subheader("🔮 30-Day Demand Forecast")
st.caption("Revenue forecasts are estimates and not guarantees.")

try:
    if daily["ds"].nunique() < 2:
        st.info("Choose a date range with at least two distinct sales days to forecast.")
    else:
        model, forecast = train_forecast(daily, periods=30)
        view = forecast.tail(min(len(forecast), 90))

        forecast_fig = px.line(
            view,
            x="ds",
            y="yhat",
            labels={"ds": "Date", "yhat": "Forecast Revenue"},
            title="Historical + Forecasted Revenue"
        )

        if "y" in view.columns:
            actual = view.dropna(subset=["y"])
            if not actual.empty:
                forecast_fig.add_scatter(
                    x=actual["ds"],
                    y=actual["y"],
                    mode="lines",
                    name="Actual Revenue"
                )

        st.plotly_chart(forecast_fig, width="stretch")

           # Forecast Accuracy Evaluation
    st.subheader("📊 Forecast Accuracy")

    if len(daily) >= 10:
        split_idx = int(len(daily) * 0.8)
        train_data = daily.iloc[:split_idx]
        test_data = daily.iloc[split_idx:].copy()

        validation_model, _ = train_forecast(train_data, periods=0)
        predictions = validation_model.predict(test_data[["ds"]])

        actual_values = test_data["y"].to_numpy()
        predicted_values = predictions["yhat"].to_numpy()
        errors = actual_values - predicted_values

        mae = abs(errors).mean()
        rmse = (errors ** 2).mean() ** 0.5
        total_actual = abs(actual_values).sum()
        wape = abs(errors).sum() / total_actual * 100 if total_actual > 0 else float("nan")

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"₹{mae:,.2f}")
        col2.metric("RMSE", f"₹{rmse:,.2f}")
        col3.metric("WAPE", f"{wape:.2f}%")

        st.caption("Accuracy evaluated on the last 20% of historical sales days.")
    else:
        st.info("At least 10 days of sales history are required.")

except Exception as exc:
    st.warning(f"Forecasting could not run: {exc}")

# Customer segmentation
st.subheader("👥 Customer Segmentation")


try:
    rfm = build_rfm(filtered)
    segmented, _ = segment_customers(rfm)

    if segmented.empty:
        st.info("Not enough customer records for segmentation.")
    else:
        summary = segmented.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean")
        ).reset_index()

        left, right = st.columns(2)

        with left:
            scatter = px.scatter(
                segmented,
                x="Frequency",
                y="Monetary",
                color="Segment",
                hover_data=["CustomerID", "Recency"],
                title="Customer Frequency vs Spend"
            )
            st.plotly_chart(scatter, width="stretch")

        with right:
            st.dataframe(summary.round(2), width="stretch")

except Exception as exc:
    st.warning(f"Customer segmentation could not run: {exc}")


# Product analytics
st.subheader("🛍️ Product Analytics")
if product_col_name:
    agg = {"Revenue": ("Revenue", "sum"), "UnitsSold": ("Quantity", "sum")}
    agg["Transactions"] = ("InvoiceNo", "nunique") if "InvoiceNo" in filtered.columns else ("Quantity", "count")
    products = filtered.groupby(product_col_name, dropna=False).agg(**agg).reset_index().sort_values("Revenue", ascending=False)
    left, right = st.columns([3, 2])
    with left:
        product_fig = px.bar(products.head(10), x=product_col_name, y="Revenue",
                             title="Top 10 Products by Revenue",
                             labels={product_col_name: "Product", "Revenue": "Revenue"})
        product_fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(product_fig, width="stretch")
    with right:
       st.dataframe(products.head(10), width="stretch")
else:
    st.info("Product analytics requires ProductName or ProductID.")

# Inventory alerts
st.subheader("📦 Inventory Alerts")
if {"ProductID", "ProductName", "StockQuantity"}.issubset(filtered.columns):
    inventory = filtered.groupby(["ProductID", "ProductName"], as_index=False).agg(
        CurrentStock=("StockQuantity", "last"), UnitsSold=("Quantity", "sum"))
    inventory["Status"] = inventory.apply(
        lambda r: "🔴 Reorder" if r["CurrentStock"] < max(20, r["UnitsSold"] / 20)
        else ("🟡 Monitor" if r["CurrentStock"] < max(40, r["UnitsSold"] / 10) else "🟢 Healthy"),
        axis=1)
    st.dataframe(inventory.sort_values("CurrentStock"), width="stretch")
else:
    st.info("Inventory alerts require ProductID, ProductName, and StockQuantity columns.")

st.caption("RetailPulse MVP • Built with Python, Scikit-learn, Prophet and Streamlit")
# Customer Churn Risk Analysis
st.subheader("🚨 Customer Churn Risk Analysis")
st.caption("Rule-based analysis using days since the customer's last purchase.")


try:
    churn_data = load_data().copy()
    churn_data["InvoiceDate"] = pd.to_datetime(churn_data["InvoiceDate"])

    reference_date = churn_data["InvoiceDate"].max()

    customer_activity = (
        churn_data.groupby("CustomerID")
        .agg(LastPurchase=("InvoiceDate", "max"),
             TotalRevenue=("Revenue", "sum"),
             Orders=("InvoiceDate", "count"))
        .reset_index()
    )

    customer_activity["DaysInactive"] = (
        reference_date - customer_activity["LastPurchase"]
    ).dt.days

    customer_activity["ChurnStatus"] = customer_activity["DaysInactive"].apply(
        lambda days: (
            "Active" if days <= 30
            else "At Risk" if days <= 90
            else "Inactive"
        )
    )

    active_count = (customer_activity["ChurnStatus"] == "Active").sum()
    risk_count = (customer_activity["ChurnStatus"] == "At Risk").sum()
    inactive_count = (customer_activity["ChurnStatus"] == "Inactive").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Customers", active_count)
    col2.metric("At-Risk Customers", risk_count)
    col3.metric("Inactive Customers", inactive_count)

    total_customers = len(customer_activity)
    churn_rate = (
        inactive_count / total_customers * 100
        if total_customers else 0
    )
    st.metric("Historical Inactivity Rate", f"{churn_rate:.2f}%")

    st.dataframe(
        customer_activity.sort_values(
            "DaysInactive", ascending=False
        ),
        width="stretch",
        hide_index=True
    )

    st.info(
        "Retention recommendation: Contact at-risk customers with "
        "personalized offers. Re-engage inactive customers with "
        "targeted win-back campaigns."
    )

except Exception as exc:
    st.warning(f"Churn analysis could not run: {exc}")
