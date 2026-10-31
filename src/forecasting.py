from prophet import Prophet

def train_forecast(daily_df, periods=30):
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(daily_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast
