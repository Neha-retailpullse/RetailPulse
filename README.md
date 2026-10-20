# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is an end-to-end retail analytics MVP that combines sales analytics,
customer segmentation, demand forecasting, and inventory alerts in an interactive
Streamlit dashboard.

## Features

- Sales and revenue KPIs
- Daily revenue trend
- 30-day demand/revenue forecasting using Prophet
- RFM customer analysis
- K-Means customer segmentation
- Inventory health/reorder alerts
- Interactive Plotly visualizations

## Project structure

RetailPulse_MVP/
├── app/
│   ├── __init__.py
│   └── dashboard.py
├── data/
│   └── retail_data.csv
├── src/
│   ├── __init__.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   └── segmentation.py
├── .gitignore
├── requirements.txt
└── README.md

## Run locally

### 1. Create environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Launch dashboard

```bash
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Then open the local Streamlit URL shown in your terminal.

## Next development stages

1. Churn prediction
2. Product-level forecasting
3. Proper inventory optimization
4. MLflow experiment tracking
5. FastAPI prediction service
6. Docker
7. Model/data drift monitoring
8. CI/CD
9. Kubernetes/cloud deployment
