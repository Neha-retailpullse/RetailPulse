
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retailpulse.db"
CSV_PATH = BASE_DIR / "data" / "retail_data.csv"


def ensure_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

        df = pd.read_csv(CSV_PATH)
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"], format="mixed", errors="coerce"
        )

        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql("sales", conn, if_exists="replace", index=False)


def load_data():
    ensure_database()

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM sales",
            conn
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="mixed", errors="coerce"
    )
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df


def add_sale(invoice_date, customer_id, product_id, quantity, region):
    if not customer_id.strip():
        raise ValueError("Please enter a Customer ID.")

    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    with sqlite3.connect(DB_PATH) as conn:
        product = conn.execute(
            """
            SELECT ProductName, Category, UnitPrice, StockQuantity
            FROM sales
            WHERE ProductID = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (product_id,)
        ).fetchone()

        if product is None:
            raise ValueError("Product not found.")

        product_name, category, unit_price, current_stock = product

        if quantity > current_stock:
            raise ValueError(
                f"Insufficient stock. Available stock: {current_stock}"
            )

        remaining_stock = current_stock - quantity
        revenue = quantity * unit_price
        formatted_date = pd.to_datetime(invoice_date).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn.execute(
            """
            INSERT INTO sales (
                InvoiceDate, CustomerID, ProductID, ProductName,
                Category, Quantity, UnitPrice, StockQuantity,
                Region, Revenue
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                formatted_date,
                customer_id.strip(),
                product_id,
                product_name,
                category,
                quantity,
                unit_price,
                remaining_stock,
                region,
                revenue
            )
        )


def daily_sales(df):
    data = df.copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )

    data["Date"] = data["InvoiceDate"].dt.date

    daily = (
        data.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .rename(columns={"Date": "ds", "Revenue": "y"})
        .sort_values("ds")
    )

    daily["ds"] = pd.to_datetime(daily["ds"])

    return daily