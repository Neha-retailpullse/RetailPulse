import pandas as pd

def load_data(path="data/retail_data.csv"):
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def daily_sales(df):
    return (
        df.groupby("InvoiceDate", as_index=False)["Revenue"]
          .sum()
          .rename(columns={"InvoiceDate": "ds", "Revenue": "y"})
          .sort_values("ds")
    )
