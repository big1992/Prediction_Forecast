import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class LSTMStrategy:
    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train(self, df, feature_cols):
        # Use Close + External Factors (Simplified for LSTM)
        # For this refactor, let's stick to the previous logic: Close + Extra
        # But to be consistent with RF, we should ideally use all features or a subset.
        # Let's use the passed feature_cols but we need to handle dimensionality carefully.
        
        data = df[feature_cols].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        seq_length = 60
        X, y = [], []
        
        # Target: Close (Index 3 in base_features? No, depends on feature_cols order)
        # We need to find index of 'Close'
        try:
            close_idx = feature_cols.index('Close')
        except ValueError:
            close_idx = 0 # Fallback
            
        for i in range(len(scaled_data)-seq_length):
            X.append(scaled_data[i:(i+seq_length)])
            y.append(scaled_data[i+seq_length, close_idx]) 
            
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, len(feature_cols))))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
        
        return model, scaler, X_test, y_test, seq_length, close_idx

    def forecast(self, model, scaler, df, days, seq_length, feature_cols, close_idx):
        last_window = df[feature_cols].iloc[-seq_length:].values
        last_window_scaled = scaler.transform(last_window)
        
        curr_seq = last_window_scaled.reshape(1, seq_length, len(feature_cols))
        future_preds = []
        
        for _ in range(days):
            pred = model.predict(curr_seq, verbose=0)[0] # Scaled Close
            future_preds.append(pred[0])
            
            # Update sequence
            last_step = curr_seq[0, -1, :].copy()
            last_step[close_idx] = pred[0] 
            
            new_step = last_step.reshape(1, 1, len(feature_cols))
            curr_seq = np.append(curr_seq[:, 1:, :], new_step, axis=1)
            
        # Inverse Transform
        dummy = np.zeros((len(future_preds), len(feature_cols)))
        dummy[:, close_idx] = future_preds
        future_preds_inv = scaler.inverse_transform(dummy)[:, close_idx]
        return future_preds_inv
