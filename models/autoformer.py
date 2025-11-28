import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism for Autoformer"""
    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Simplified auto-correlation
        queries = queries.mean(dim=2)  # [B, L, E]
        keys = keys.mean(dim=2)  # [B, S, E]
        
        # Compute correlation
        correlation = torch.matmul(queries, keys.transpose(-2, -1))  # [B, L, S]
        correlation = torch.softmax(correlation / np.sqrt(E), dim=-1)
        
        values = values.mean(dim=2)  # [B, S, D]
        out = torch.matmul(correlation, values)  # [B, L, D]
        
        return out.unsqueeze(2)  # [B, L, 1, D]

class AutoformerModel(nn.Module):
    """Simplified Autoformer for stock prediction"""
    def __init__(self, seq_len, pred_len, enc_in, d_model=64, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.autocorr = AutoCorrelation()
        self.decomp = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [B, seq_len, enc_in]
        B, L, _ = x.shape
        x = self.enc_embedding(x)  # [B, L, d_model]
        
        # Decomposition
        x_mean = self.decomp(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_mean
        
        # Auto-correlation on seasonal
        x_seasonal = x_seasonal.unsqueeze(2)  # [B, L, 1, d_model]
        x_out = self.autocorr(x_seasonal, x_seasonal, x_seasonal)
        x_out = x_out.squeeze(2) + x_mean
        
        # Project
        x_out = self.projection(x_out[:, -self.pred_len:, :])
        return x_out.squeeze(-1)

class AutoformerStrategy:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, df, feature_cols, seq_len=60, pred_len=1):
        data = df[feature_cols].values
        target = df['Target'].values
        
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - seq_len - pred_len + 1):
            X.append(data_scaled[i:i+seq_len])
            y.append(target[i+seq_len:i+seq_len+pred_len])
        
        X = np.array(X)
        y = np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        
        model = AutoformerModel(seq_len, pred_len, len(feature_cols)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train.squeeze())
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model, scaler, X_test, y_test, seq_len, feature_cols
    
    def forecast(self, model, scaler, df, days, seq_len, feature_cols):
        model.eval()
        last_seq = df[feature_cols].iloc[-seq_len:].values
        last_seq_scaled = scaler.transform(last_seq)
        
        predictions = []
        current_seq = torch.FloatTensor(last_seq_scaled).unsqueeze(0).to(self.device)
        
        for _ in range(days):
            with torch.no_grad():
                pred = model(current_seq)
                pred_value = pred[0, -1].item()
                predictions.append(pred_value)
                
                new_step = current_seq[0, -1, :].clone()
                try:
                    close_idx = feature_cols.index('Close')
                    new_step[close_idx] = pred_value
                except:
                    new_step[3] = pred_value
                    
                current_seq = torch.cat([current_seq[:, 1:, :], new_step.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return predictions
