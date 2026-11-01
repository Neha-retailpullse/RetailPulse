# RetailPulse - AI-Powered Customer Analytics and Demand Forecasting

RetailPulse is an end-to-end retail analytics MVP that combines sales analytics, customer segmentation, demand forecasting, customer inactivity analysis, and inventory alerts in an interactive Streamlit dashboard.

## Live Dashboard

[Open RetailPulse Dashboard](https://retailpulse-neha.streamlit.app)

## Features

- Sales Analytics: Revenue KPIs, daily sales trends, and business performance insights.
- Customer Analytics: Customer purchase behaviour and RFM analysis.
- RFM Segmentation: Recency, Frequency, and Monetary analysis using K-Means clustering.
- Customer Churn Analysis: Rule-based identification of active, at-risk, and inactive customers.
- Demand Forecasting: 30-day revenue forecasting using Prophet.
- Forecast Evaluation: MAE, RMSE, and WAPE metrics.
- Inventory Analysis: Inventory health and reorder alerts.
- Add New Sale: Interactive form to record new sales.
- Interactive Visualizations: Plotly charts and Streamlit dashboard components.

## Project Architecture

1. Data Layer: Retail transaction data.
2. Feature Engineering: RFM metrics and sales-related features.
3. Model Layer: Prophet forecasting and K-Means customer segmentation.
4. Serving Layer: Streamlit interactive dashboard.
5. Monitoring Layer: Future scope for model performance and data drift monitoring.

## Project Structure

```text
RetailPulse_MVP/
|-- app/
|   |-- __init__.py
|   |-- dashboard.py
|-- data/
|   |-- retail_data.csv
|-- src/
|   |-- __init__.py
|   |-- forecasting.py
|   |-- preprocessing.py
|   |-- segmentation.py
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Prophet
- Streamlit
- Plotly
- Git and GitHub

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Neha-retailpullse/RetailPulse.git
cd RetailPulse
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the dashboard

```powershell
python -m streamlit run app/dashboard.py
```

Open the local URL displayed in your terminal.

## Project Workflow

1. Dataset collection
2. Library imports and data loading
3. Data cleaning and preprocessing
4. Feature engineering
5. Exploratory Data Analysis
6. RFM customer segmentation
7. Demand forecasting using time-series analysis
8. Customer inactivity analysis
9. Inventory health analysis
10. Interactive dashboard development
11. Streamlit deployment

## Future Enhancements

- Advanced machine-learning churn prediction
- Product-level demand forecasting
- Advanced inventory optimization
- MLflow experiment tracking
- FastAPI prediction service
- Docker containerization
- Model and data drift monitoring
- CI/CD automation
- Kubernetes and cloud deployment

## Project Information

**Project:** RetailPulse - AI-Powered Customer Analytics and Demand Forecasting

**GitHub Repository:** [RetailPulse](https://github.com/Neha-retailpullse/RetailPulse)

**Live Deployment:** [Streamlit Cloud](https://retailpulse-neha.streamlit.app)