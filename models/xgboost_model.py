import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class XGBoostStrategy:
    def train(self, df, feature_cols, model_type='xgboost'):
        X = df[feature_cols]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scaling is less critical for Trees but good for consistency if mixing
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, early_stopping_rounds=50)
            # XGBoost requires eval set for early stopping
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        else:
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(50)])
            
        return model, scaler, X_test, y_test, X_test_scaled

    def forecast(self, model, scaler, last_data, days, feature_cols):
        future_preds = []
        current_input_raw = last_data[feature_cols].values.reshape(1, -1).copy()
        
        # Indices for OHLC update (assuming they are present in feature_cols)
        # We need to be careful if feature_cols order changes.
        # Let's assume OHLC are in the list.
        try:
            open_idx = feature_cols.index('Open')
            high_idx = feature_cols.index('High')
            low_idx = feature_cols.index('Low')
            close_idx = feature_cols.index('Close')
        except:
            # Fallback if OHLC not found (e.g. only derived features)
            # This naive update won't work well if we don't have base prices.
            # For now, assume they exist.
            open_idx, high_idx, low_idx, close_idx = 0, 1, 2, 3
            
        for _ in range(days):
            current_input_scaled = scaler.transform(current_input_raw)
            pred = model.predict(current_input_scaled)[0]
            future_preds.append(pred)
            
            # Naive Update
            current_input_raw[0, open_idx] = pred
            current_input_raw[0, high_idx] = pred
            current_input_raw[0, low_idx] = pred
            current_input_raw[0, close_idx] = pred
            
        return future_preds
