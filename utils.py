import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from fredapi import Fred
from datetime import date, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
GLOBAL_MARKETS = {
    "VIX": "^VIX",
    "S&P 500 Futures": "ES=F",
    "Nasdaq Futures": "NQ=F",
    "Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "SET Index": "^SET.BK"
}

MACRO_SERIES = {
    "10Y Bond Yield": "DGS10",
    "CPI": "CPIAUCSL",
    "Fed Funds Rate": "FEDFUNDS",
    "Unemployment Rate": "UNRATE"
}

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_main_ticker(ticker, start, end):
    """Load the main stock data."""
    try:
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Configure custom session to avoid impersonation issues
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Try multiple methods
        df = pd.DataFrame()
        
        # Method 1: Ticker.history with custom session (if supported)
        try:
            dat = yf.Ticker(ticker, session=session)
            df = dat.history(start=start, end=end, auto_adjust=False)
        except Exception:
            pass
            
        # Method 2: Standard download with ignore_tz
        if df.empty:
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, ignore_tz=True)
            except Exception:
                pass
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        
        # Fix timezone issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

@st.cache_data
def load_global_markets(start, end):
    """Load global market indices and commodities."""
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    market_data = {}
    for name, symbol in GLOBAL_MARKETS.items():
        try:
            dat = yf.Ticker(symbol)
            df = dat.history(start=start, end=end, auto_adjust=False)
            
            if df.empty:
                df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
                
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                market_data[name] = df['Close']
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return market_data

@st.cache_data
def load_macro_data(api_key, start, end):
    """Load macro data from FRED API."""
    if not api_key:
        return {}
    
    fred = Fred(api_key=api_key)
    macro_data = {}
    
    for name, series_id in MACRO_SERIES.items():
        try:
            # FRED data is often monthly/daily, need to align later
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            macro_data[name] = series
        except Exception as e:
            st.warning(f"Could not load {name} from FRED: {e}")
            
    return macro_data

def merge_data(main_df, market_data, macro_data):
    """Merge all external data sources into the main dataframe."""
    df = main_df.copy()
    
    # Merge Global Markets (Yahoo)
    for name, series in market_data.items():
        series.name = f"Ext_{name.replace(' ', '_')}"
        df = df.join(series, how='left')
        df[series.name] = df[series.name].ffill()
        
    # Merge Macro Data (FRED)
    for name, series in macro_data.items():
        series.name = f"Macro_{name.replace(' ', '_')}"
        # FRED data might have different index (datetime), join logic is same
        df = df.join(series, how='left')
        df[series.name] = df[series.name].ffill()
        
    return df.dropna()

# ==========================================
# FEATURE ENGINEERING (QUANT)
# ==========================================
def add_quant_features(df):
    """Add advanced technical and quantitative features."""
    df = df.copy()
    # Ensure no NaNs before calculation if possible, or handle after
    
    # 1. Basic Trend
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    
    # 2. Momentum
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # 3. Volatility (New)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None and not bb.empty:
        # Column names might vary, find them dynamically
        bb_cols = bb.columns.tolist()
        upper_col = [c for c in bb_cols if 'BBU' in c or 'upper' in c.lower()]
        lower_col = [c for c in bb_cols if 'BBL' in c or 'lower' in c.lower()]
        mid_col = [c for c in bb_cols if 'BBM' in c or 'mid' in c.lower()]
        
        if upper_col and lower_col and mid_col:
            df['BB_Width'] = (bb[upper_col[0]] - bb[lower_col[0]]) / bb[mid_col[0]]
        else:
            # Fallback: just skip BB_Width if columns not found
            pass
    
    # Historical Volatility
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # 4. Volume Signals (New)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['Vol_SMA_20'] = ta.sma(df['Volume'], length=20)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    
    # 5. Seasonality (New)
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
    
    # Target
    df['Target'] = df['Close'].shift(-1)
    
    return df.dropna()

def prepare_features(df):
    """Get list of all numerical features for the model."""
    exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude_cols]
    # Include OHLCV if needed, but usually we use derived features. 
    # However, RF/XGBoost benefit from raw prices too sometimes, but scaling is key.
    # Let's include OHLCV for now as base.
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    return base_features + features
