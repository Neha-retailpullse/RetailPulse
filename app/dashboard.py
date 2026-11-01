
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import load_data, daily_sales, add_sale
from src.segmentation import build_rfm, segment_customers
from src.forecasting import train_forecast


st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #F4F7FC; }
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 { color: #17365D !important; font-weight: 800 !important; }
h2, h3 { color: #24527A !important; }
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE6F1;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")


# Load and prepare data
try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("The dataset is empty.")
    st.stop()

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


# Sidebar navigation
pages = [
    "1. Business Overview",
    "2. Sales Analytics",
    "3. Revenue Analysis",
    "4. Product Analytics",
    "5. Category Analytics",
    "6. Regional Analytics",
    "7. Customer Overview",
    "8. Customer Segmentation",
    "9. Churn Analysis",
    "10. Demand Forecasting",
    "11. Inventory Management",
    "12. Add New Sale"
]

st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("Analytics Navigation")

page = st.sidebar.radio("Go to", pages)

st.sidebar.divider()
st.sidebar.subheader("Dashboard Filters")

date_min = df["InvoiceDate"].min().date()
date_max = df["InvoiceDate"].max().date()

selected_dates = st.sidebar.date_input(
    "Invoice date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

product_col_name = (
    "ProductName" if "ProductName" in df.columns
    else "ProductID" if "ProductID" in df.columns
    else None
)

product_options = (
    sorted(df[product_col_name].dropna().astype(str).unique())
    if product_col_name else []
)

selected_products = st.sidebar.multiselect(
    "Products (empty = all)",
    product_options
)

filtered = df.copy()

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    filtered = filtered[
        filtered["InvoiceDate"].dt.date.between(
            selected_dates[0], selected_dates[1]
        )
    ]

if product_col_name and selected_products:
    filtered = filtered[
        filtered[product_col_name].astype(str).isin(selected_products)
    ]

if filtered.empty:
    st.warning("No data matches these filters.")
    st.stop()


# Common calculations
revenue = filtered["Revenue"].sum()
transactions = (
    filtered["InvoiceNo"].nunique()
    if "InvoiceNo" in filtered.columns
    else len(filtered)
)
customers = filtered["CustomerID"].nunique()
units = filtered["Quantity"].sum()
avg_transaction = revenue / transactions if transactions else 0

daily = daily_sales(filtered)
daily["ds"] = pd.to_datetime(daily["ds"])
daily = (
    daily.sort_values("ds")
    .set_index("ds")
    .asfreq("D", fill_value=0)
    .rename_axis("ds")
    .reset_index()
)


# PAGE 1 - BUSINESS OVERVIEW
if page == pages[0]:

    st.header("📌 Business Overview")

    cols = st.columns(5)
    cols[0].metric("Total Revenue", f"₹{revenue:,.0f}")
    cols[1].metric("Transactions", f"{transactions:,}")
    cols[2].metric("Customers", f"{customers:,}")
    cols[3].metric("Units Sold", f"{units:,.0f}")
    cols[4].metric("Average Transaction", f"₹{avg_transaction:,.0f}")

    st.divider()
    st.subheader("💡 Sales Insights")

    if product_col_name:
        product_revenue = (
            filtered.groupby(product_col_name)["Revenue"]
            .sum().sort_values(ascending=False)
        )
        top_product = product_revenue.index[0]
    else:
        top_product = "N/A"

    best_day = daily.loc[daily["y"].idxmax()] if not daily.empty else None

    a, b, c = st.columns(3)
    a.metric("Top Product", str(top_product))
    b.metric(
        "Highest Revenue Day",
        pd.to_datetime(best_day["ds"]).strftime("%d %b %Y")
        if best_day is not None else "N/A"
    )
    c.metric(
        "Unique Products",
        filtered[product_col_name].nunique()
        if product_col_name else "N/A"
    )

    st.subheader("📈 Daily Revenue Trend")
    st.plotly_chart(
        px.line(daily, x="ds", y="y", markers=True,
                title="Daily Revenue"),
        width="stretch"
    )


# PAGE 2 - SALES ANALYTICS
elif page == pages[1]:

    st.header("📈 Sales Analytics")

    a, b, c = st.columns(3)
    a.metric("Transactions", f"{transactions:,}")
    b.metric("Units Sold", f"{units:,.0f}")
    c.metric("Average Transaction", f"₹{avg_transaction:,.0f}")

    st.plotly_chart(
        px.line(daily, x="ds", y="y", markers=True,
                title="Daily Sales Performance"),
        width="stretch"
    )

    st.plotly_chart(
        px.bar(daily, x="ds", y="y", title="Daily Revenue Comparison"),
        width="stretch"
    )


# PAGE 3 - REVENUE ANALYSIS
elif page == pages[2]:

    st.header("💰 Revenue Analysis")

    a, b = st.columns(2)
    a.metric("Total Revenue", f"₹{revenue:,.2f}")
    b.metric("Average Transaction Value", f"₹{avg_transaction:,.2f}")

    monthly = (
        filtered.set_index("InvoiceDate")["Revenue"]
        .resample("ME").sum().reset_index()
    )

    st.plotly_chart(
        px.line(monthly, x="InvoiceDate", y="Revenue",
                markers=True, title="Monthly Revenue Trend"),
        width="stretch"
    )

    st.plotly_chart(
        px.histogram(filtered, x="Revenue", nbins=30,
                     title="Transaction Revenue Distribution"),
        width="stretch"
    )


# PAGE 4 - PRODUCT ANALYTICS
elif page == pages[3]:

    st.header("🛍️ Product Analytics")

    if product_col_name:
        products = (
            filtered.groupby(product_col_name)
            .agg(
                Revenue=("Revenue", "sum"),
                UnitsSold=("Quantity", "sum")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(products.head(10), x=product_col_name, y="Revenue",
                   title="Top 10 Products by Revenue"),
            width="stretch"
        )

        st.plotly_chart(
            px.bar(products.sort_values("Revenue").head(10),
                   x=product_col_name, y="Revenue",
                   title="Lowest 10 Products by Revenue"),
            width="stretch"
        )

        st.subheader("Product Performance Table")
        st.dataframe(products, width="stretch", hide_index=True)
    else:
        st.info("Product information is unavailable.")


# PAGE 5 - CATEGORY ANALYTICS
elif page == pages[4]:

    st.header("📦 Category Analytics")

    if "Category" in filtered.columns:
        categories = (
            filtered.groupby("Category")
            .agg(
                Revenue=("Revenue", "sum"),
                UnitsSold=("Quantity", "sum")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(categories, x="Category", y="Revenue",
                   title="Revenue by Category"),
            width="stretch"
        )

        st.plotly_chart(
            px.pie(categories, names="Category", values="Revenue",
                   title="Category Revenue Share"),
            width="stretch"
        )

        st.dataframe(categories, width="stretch", hide_index=True)
    else:
        st.info("Category column is unavailable.")


# PAGE 6 - REGIONAL ANALYTICS
elif page == pages[5]:

    st.header("🌍 Regional Analytics")

    if "Region" in filtered.columns:
        regions = (
            filtered.groupby("Region")
            .agg(
                Revenue=("Revenue", "sum"),
                UnitsSold=("Quantity", "sum"),
                Customers=("CustomerID", "nunique")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(regions, x="Region", y="Revenue",
                   title="Revenue by Region"),
            width="stretch"
        )

        st.plotly_chart(
            px.pie(regions, names="Region", values="Revenue",
                   title="Regional Revenue Share"),
            width="stretch"
        )

        st.dataframe(regions, width="stretch", hide_index=True)
    else:
        st.info("Region column is unavailable.")


# PAGE 7 - CUSTOMER OVERVIEW
elif page == pages[6]:

    st.header("👥 Customer Overview")

    customer_data = (
        filtered.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceDate", "count"),
            Units=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    a, b, c = st.columns(3)
    a.metric("Total Customers", len(customer_data))
    b.metric("Average Customer Revenue",
             f"₹{customer_data['Revenue'].mean():,.2f}")
    c.metric("Average Orders",
             f"{customer_data['Orders'].mean():.2f}")

    st.plotly_chart(
        px.histogram(customer_data, x="Revenue", nbins=30,
                     title="Customer Spending Distribution"),
        width="stretch"
    )

    st.plotly_chart(
        px.scatter(customer_data, x="Orders", y="Revenue",
                   title="Customer Orders vs Revenue"),
        width="stretch"
    )

    st.dataframe(customer_data, width="stretch", hide_index=True)


# PAGE 8 - CUSTOMER SEGMENTATION
elif page == pages[7]:

    st.header("👥 Customer Segmentation")

    try:
        rfm = build_rfm(filtered)
        segmented, _ = segment_customers(rfm)

        if not segmented.empty:
            summary = segmented.groupby("Segment").agg(
                Customers=("CustomerID", "count"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean")
            ).reset_index()

            st.plotly_chart(
                px.scatter(
                    segmented,
                    x="Frequency",
                    y="Monetary",
                    color="Segment",
                    hover_data=["CustomerID", "Recency"],
                    title="Customer Frequency vs Spending"
                ),
                width="stretch"
            )

            st.subheader("Customer Segment Summary")
            st.dataframe(summary.round(2), width="stretch",
                         hide_index=True)
        else:
            st.info("Not enough customers for segmentation.")

    except Exception as exc:
        st.warning(f"Segmentation could not run: {exc}")


# PAGE 9 - CHURN ANALYSIS
elif page == pages[8]:

    st.header("🚨 Customer Churn Risk Analysis")
    st.caption("Rule-based analysis based on days since last purchase.")

    churn_data = df.copy()
    reference_date = churn_data["InvoiceDate"].max()

    activity = (
        churn_data.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            TotalRevenue=("Revenue", "sum"),
            Orders=("InvoiceDate", "count")
        )
        .reset_index()
    )

    activity["DaysInactive"] = (
        reference_date - activity["LastPurchase"]
    ).dt.days

    activity["ChurnStatus"] = activity["DaysInactive"].apply(
        lambda days: "Active" if days <= 30
        else "At Risk" if days <= 90
        else "Inactive"
    )

    a, b, c = st.columns(3)
    a.metric("Active", (activity["ChurnStatus"] == "Active").sum())
    b.metric("At Risk", (activity["ChurnStatus"] == "At Risk").sum())
    c.metric("Inactive", (activity["ChurnStatus"] == "Inactive").sum())

    st.plotly_chart(
        px.histogram(activity, x="DaysInactive", color="ChurnStatus",
                     title="Customer Inactivity Distribution"),
        width="stretch"
    )

    st.dataframe(
        activity.sort_values("DaysInactive", ascending=False),
        width="stretch",
        hide_index=True
    )

    st.info("Contact at-risk customers with personalized retention offers.")


# PAGE 10 - DEMAND FORECASTING
elif page == pages[9]:

    st.header("🔮 Demand Forecasting")
    st.caption("Revenue forecasts are estimates, not guarantees.")

    if daily["ds"].nunique() < 2:
        st.info("At least two distinct sales days are needed.")
    else:
        try:
            model, forecast = train_forecast(daily, periods=30)

            view = forecast.tail(min(len(forecast), 90))

            fig = px.line(
                view, x="ds", y="yhat",
                title="Historical and Forecasted Revenue"
            )

            if "y" in view.columns:
                actual = view.dropna(subset=["y"])
                if not actual.empty:
                    fig.add_scatter(
                        x=actual["ds"], y=actual["y"],
                        mode="lines", name="Actual Revenue"
                    )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Forecast Accuracy")

            if len(daily) >= 10:
                split_idx = int(len(daily) * 0.8)
                train_data = daily.iloc[:split_idx]
                test_data = daily.iloc[split_idx:].copy()

                validation_model, _ = train_forecast(
                    train_data, periods=0
                )
                predictions = validation_model.predict(test_data[["ds"]])

                errors = (
                    test_data["y"].to_numpy()
                    - predictions["yhat"].to_numpy()
                )

                mae = abs(errors).mean()
                rmse = (errors ** 2).mean() ** 0.5
                total_actual = abs(test_data["y"].to_numpy()).sum()
                wape = (
                    abs(errors).sum() / total_actual * 100
                    if total_actual > 0 else float("nan")
                )

                a, b, c = st.columns(3)
                a.metric("MAE", f"₹{mae:,.2f}")
                b.metric("RMSE", f"₹{rmse:,.2f}")
                c.metric("WAPE", f"{wape:.2f}%")

                st.caption("Evaluation uses the last 20% of historical sales days.")

        except Exception as exc:
            st.warning(f"Forecasting could not run: {exc}")


# PAGE 11 - INVENTORY MANAGEMENT
elif page == pages[10]:

    st.header("📦 Inventory Management")

    required_inventory = {"ProductID", "ProductName", "StockQuantity"}

    if required_inventory.issubset(filtered.columns):

        inventory = (
            filtered.groupby(["ProductID", "ProductName"], as_index=False)
            .agg(
                CurrentStock=("StockQuantity", "last"),
                UnitsSold=("Quantity", "sum")
            )
        )

        inventory["Status"] = inventory.apply(
            lambda r: (
                "Reorder" if r["CurrentStock"] < max(20, r["UnitsSold"] / 20)
                else "Monitor" if r["CurrentStock"] < max(40, r["UnitsSold"] / 10)
                else "Healthy"
            ),
            axis=1
        )

        a, b, c = st.columns(3)
        a.metric("Products", len(inventory))
        b.metric("Reorder Required", (inventory["Status"] == "Reorder").sum())
        c.metric("Healthy Stock", (inventory["Status"] == "Healthy").sum())

        st.plotly_chart(
            px.bar(inventory, x="ProductName", y="CurrentStock",
                   color="Status", title="Current Stock by Product"),
            width="stretch"
        )

        st.subheader("Inventory Status Table")
        st.dataframe(
            inventory.sort_values("CurrentStock"),
            width="stretch",
            hide_index=True
        )

    else:
        st.info("Required inventory columns are unavailable.")


# PAGE 12 - ADD NEW SALE
elif page == pages[11]:

    st.header("➕ Add New Sale")
    st.caption("Record a new retail transaction.")

    products_df = load_data()

    product_options_df = products_df[
        ["ProductID", "ProductName"]
    ].drop_duplicates()

    with st.form("add_sale_form"):

        selected_product = st.selectbox(
            "Select Product",
            product_options_df["ProductID"].tolist(),
            format_func=lambda x: (
                f"{x} - "
                f"{product_options_df.loc[product_options_df['ProductID'] == x, 'ProductName'].iloc[0]}"
            )
        )

        customer_id = st.text_input("Customer ID")
        sale_date = st.date_input("Sale Date")
        quantity = st.number_input("Quantity", min_value=1, step=1)

        region_options = sorted(
            products_df["Region"].dropna().unique()
        )
        region = st.selectbox("Region", region_options)

        submitted = st.form_submit_button("Save Sale")

        if submitted:
            if not customer_id.strip():
                st.error("Please enter a Customer ID.")
            else:
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


st.sidebar.divider()
st.sidebar.caption("RetailPulse MVP | Built with Python and Streamlit")
st.caption("RetailPulse MVP | Customer Analytics & Demand Forecasting")
