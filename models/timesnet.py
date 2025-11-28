import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimesBlock(nn.Module):
    """Simplified TimesNet block using 1D convolution"""
    def __init__(self, d_model, d_ff, num_kernels=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=num_kernels, padding=num_kernels//2),
            nn.GELU(),
            nn.Conv1d(d_ff, d_model, kernel_size=num_kernels, padding=num_kernels//2)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Apply 1D convolution
        x_out = x.permute(0, 2, 1)  # [B, D, T]
        x_out = self.conv(x_out)     # [B, D, T']
        x_out = x_out.permute(0, 2, 1)  # [B, T', D]
        
        # Handle size mismatch for residual
        if x_out.shape[1] != T:
            x_out = x_out[:, :T, :]  # Trim to match
        
        return x_out

class TimesNetModel(nn.Module):
    """Simplified TimesNet for stock prediction"""
    def __init__(self, seq_len, pred_len, enc_in, d_model=64, d_ff=128, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.layers = nn.ModuleList([TimesBlock(d_model, d_ff) for _ in range(num_layers)])
        self.projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [B, seq_len, enc_in]
        x = self.enc_embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual
        x = self.projection(x[:, -self.pred_len:, :])
        return x.squeeze(-1)

class TimesNetStrategy:
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
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        
        # Model
        model = TimesNetModel(seq_len, pred_len, len(feature_cols)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(10):  # Reduced for demo
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
                
                # Update sequence (naive)
                # This is simplified - in practice you'd need to inverse transform and update features
                new_step = current_seq[0, -1, :].clone()
                new_step[feature_cols.index('Close')] = pred_value
                current_seq = torch.cat([current_seq[:, 1:, :], new_step.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return predictions
