import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BacktestEngine:
    def __init__(self, model_strategy):
        self.strategy = model_strategy

    def walk_forward_validation(self, df, feature_cols, train_window=252, test_window=20, steps=5):
        """
        Perform Walk-Forward Validation.
        train_window: Number of days to train on.
        test_window: Number of days to predict.
        steps: Number of walk-forward steps.
        """
        metrics = []
        total_len = len(df)
        
        # Start index for the first fold
        start_idx = total_len - (steps * test_window) - train_window
        if start_idx < 0:
            start_idx = 0
            
        for i in range(steps):
            # Define windows
            train_start = start_idx + (i * test_window)
            train_end = train_start + train_window
            test_end = train_end + test_window
            
            if test_end > total_len:
                break
                
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[train_end:test_end]
            
            # Train
            try:
                # Note: strategy.train returns tuple, we need model and scaler
                # Assuming standard signature for all strategies
                model, scaler, _, _, _ = self.strategy.train(train_df, feature_cols)
                
                # Predict
                # We need to handle different model input requirements (scaled vs raw)
                # Most of our strategies return X_test_scaled in the train output, but here we have a fresh test_df
                X_test = test_df[feature_cols]
                X_test_scaled = scaler.transform(X_test)
                
                # Handle Hybrid/LSTM sequence requirements if necessary
                # For simplicity, this generic engine assumes standard sklearn-like predict interface on scaled data
                # If Hybrid, we might need special handling. 
                # Let's assume the strategy object has a 'predict_for_backtest' or we use the model directly.
                
                if hasattr(model, 'predict'):
                    preds = model.predict(X_test_scaled)
                else:
                    # Fallback or error
                    continue
                    
                # Calculate Metrics
                y_true = test_df['Target'].values # Target is next close
                # Wait, Target is shifted. The last row of test_df has NaN Target?
                # We should dropna in utils, so test_df should have valid Targets.
                
                mae = mean_absolute_error(y_true, preds)
                rmse = np.sqrt(mean_squared_error(y_true, preds))
                
                metrics.append({
                    "Fold": i+1,
                    "MAE": mae,
                    "RMSE": rmse
                })
            except Exception as e:
                print(f"Backtest error fold {i}: {e}")
                
        return pd.DataFrame(metrics)

    def test_strategy(self, df, signals):
        """
        Simple strategy backtest based on Buy/Sell signals.
        signals: Series of 1 (Buy), -1 (Sell), 0 (Hold)
        """
        df = df.copy()
        df['Signal'] = signals
        df['Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
        
        df['Cum_Return'] = (1 + df['Strategy_Return']).cumprod()
        
        total_return = df['Cum_Return'].iloc[-1] - 1
        sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)
        
        # Max Drawdown
        roll_max = df['Cum_Return'].cummax()
        drawdown = df['Cum_Return'] / roll_max - 1
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }
