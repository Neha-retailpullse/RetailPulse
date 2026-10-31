
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "retail_data.csv"
DB_PATH = BASE_DIR / "database" / "retailpulse.db"

def setup_database():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, parse_dates=["InvoiceDate"])

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("sales", conn, if_exists="replace", index=False)

    print(f"Database created successfully: {DB_PATH}")
    print(f"Records imported: {len(df)}")

if __name__ == "__main__":
    setup_database()
