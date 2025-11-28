import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class HybridStrategy:
    def train(self, df, feature_cols):
        # 1. Prepare Data
        X = df[feature_cols]
        y = df['Target']
        
        # Split
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Scale
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. Train LSTM (Short-term patterns)
        # Reshape for LSTM [samples, time steps, features]
        # We use a sliding window of 1 for simplicity in this hybrid architecture 
        # (or we could use sequence generation like in pure LSTM, but that complicates alignment with XGB)
        # Let's use sequence generation but we need to align y.
        
        seq_length = 10 # Shorter sequence for hybrid
        X_train_lstm, y_train_lstm = self._create_sequences(X_train_scaled, y_train.values, seq_length)
        X_test_lstm, y_test_lstm = self._create_sequences(X_test_scaled, y_test.values, seq_length)
        
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=False, input_shape=(seq_length, len(feature_cols))))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)
        
        # 3. Get LSTM Predictions for Training Set (to feed into XGBoost)
        lstm_pred_train = lstm_model.predict(X_train_lstm, verbose=0)
        
        # 4. Train XGBoost on Residuals or as Meta-Learner
        # Approach: Weighted Average or Stacking.
        # Let's do Stacking: XGBoost takes Original Features + LSTM Prediction
        
        # We need to align X_train for XGBoost with LSTM output (trimmed by seq_length)
        X_train_xgb = X_train_scaled[seq_length:]
        y_train_xgb = y_train.values[seq_length:]
        
        # Add LSTM pred as feature
        X_train_combined = np.hstack((X_train_xgb, lstm_pred_train))
        
        xgb_model = xgb.XGBRegressor(n_estimators=100)
        xgb_model.fit(X_train_combined, y_train_xgb)
        
        return lstm_model, xgb_model, scaler, X_test_scaled, y_test.values[seq_length:], seq_length

    def _create_sequences(self, X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X)-seq_length):
            xs.append(X[i:(i+seq_length)])
            ys.append(y[i+seq_length])
        return np.array(xs), np.array(ys)

    def forecast(self, lstm_model, xgb_model, scaler, df, days, seq_length, feature_cols):
        # Prepare initial sequence
        last_window = df[feature_cols].iloc[-seq_length:].values
        last_window_scaled = scaler.transform(last_window)
        
        curr_seq = last_window_scaled.reshape(1, seq_length, len(feature_cols))
        future_preds = []
        
        # Indices for OHLC update
        try:
            close_idx = feature_cols.index('Close')
        except:
            close_idx = 0
            
        for _ in range(days):
            # 1. LSTM Predict
            lstm_pred = lstm_model.predict(curr_seq, verbose=0)
            
            # 2. XGBoost Predict
            # Input: Current Step Features (Last row of seq) + LSTM Pred
            current_features = curr_seq[0, -1, :].reshape(1, -1)
            combined_input = np.hstack((current_features, lstm_pred))
            
            final_pred = xgb_model.predict(combined_input)[0]
            future_preds.append(final_pred)
            
            # 3. Update Sequence
            last_step = curr_seq[0, -1, :].copy()
            last_step[close_idx] = final_pred 
            
            new_step = last_step.reshape(1, 1, len(feature_cols))
            curr_seq = np.append(curr_seq[:, 1:, :], new_step, axis=1)
            
        return future_preds
