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

        df = pd.read_csv(CSV_PATH, parse_dates=["InvoiceDate"])

        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql("sales", conn, if_exists="replace", index=False)


def load_data():
    ensure_database()

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM sales", conn)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df


def daily_sales(df):
    return (
        df.groupby("InvoiceDate", as_index=False)["Revenue"]
          .sum()
          .rename(columns={"InvoiceDate": "ds", "Revenue": "y"})
          .sort_values("ds")
    )