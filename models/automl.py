from sklearn.metrics import mean_squared_error
import numpy as np
from models.random_forest import RandomForestStrategy
from models.xgboost_model import XGBoostStrategy
# Import LSTM/Hybrid if needed, but they are slower for AutoML loop
# Let's stick to fast models for AutoML: RF, XGB, LGBM

class AutoMLSelector:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestStrategy(),
            "XGBoost": XGBoostStrategy(),
            "LightGBM": XGBoostStrategy() # Uses 'lightgbm' param
        }
        self.best_model_name = None
        self.best_model_instance = None
        self.best_scaler = None
        self.best_rmse = float('inf')

    def find_best_model(self, df, feature_cols):
        best_res = None
        
        for name, strategy in self.models.items():
            try:
                # Train
                if name == "LightGBM":
                    model, scaler, X_test, y_test, X_test_scaled = strategy.train(df, feature_cols, model_type='lightgbm')
                else:
                    model, scaler, X_test, y_test, X_test_scaled = strategy.train(df, feature_cols)
                
                # Evaluate
                preds = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_model_name = name
                    self.best_model_instance = model
                    self.best_scaler = scaler
                    best_res = (model, scaler, X_test, y_test, X_test_scaled)
                    
            except Exception as e:
                print(f"AutoML failed for {name}: {e}")
                continue
                
        return self.best_model_name, self.best_rmse, best_res
