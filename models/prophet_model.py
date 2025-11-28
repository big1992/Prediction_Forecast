from prophet import Prophet
import pandas as pd

class ProphetStrategy:
    def train(self, df, extra_features):
        prophet_df = df.reset_index()[['Date', 'Close'] + extra_features]
        prophet_df.columns = ['ds', 'y'] + extra_features
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        model = Prophet()
        for feat in extra_features:
            model.add_regressor(feat)
        model.fit(prophet_df)
        return model

    def forecast(self, model, df, days, extra_features):
        future = model.make_future_dataframe(periods=days)
        
        # Fill regressors
        df_lookup = df.copy()
        df_lookup.index = df_lookup.index.tz_localize(None)
        
        for feat in extra_features:
            future[feat] = future['ds'].map(df_lookup[feat])
            future[feat] = future[feat].ffill() # Naive future fill
            
        forecast = model.predict(future)
        
        future_part = forecast.iloc[-days:]
        return future_part['yhat'].values, future_part['ds'].values, forecast
