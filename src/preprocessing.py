
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retailpulse.db"

def load_data():
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
