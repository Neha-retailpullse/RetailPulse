import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def build_rfm(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceDate", "count"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    return rfm

def segment_customers(rfm, n_clusters=4):
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Monetary"] = features["Monetary"].clip(lower=0)
    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(X)
    return rfm, model
