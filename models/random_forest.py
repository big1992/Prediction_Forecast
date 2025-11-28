from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class RandomForestStrategy:
    def train(self, df, feature_cols):
        X = df[feature_cols]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, X_test, y_test, X_test_scaled

    def forecast(self, model, scaler, last_data, days, feature_cols):
        future_preds = []
        current_input_raw = last_data[feature_cols].values.reshape(1, -1).copy()
        
        for _ in range(days):
            current_input_scaled = scaler.transform(current_input_raw)
            pred = model.predict(current_input_scaled)[0]
            future_preds.append(pred)
            
            # Naive Update for OHLC
            current_input_raw[0, 0] = pred # Open
            current_input_raw[0, 1] = pred # High
            current_input_raw[0, 2] = pred # Low
            current_input_raw[0, 3] = pred # Close
            
        return future_preds
