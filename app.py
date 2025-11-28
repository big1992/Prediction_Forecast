import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Custom Modules
import utils
from models.random_forest import RandomForestStrategy
from models.lstm import LSTMStrategy
from models.prophet_model import ProphetStrategy
from models.xgboost_model import XGBoostStrategy
from models.hybrid_model import HybridStrategy
from models.automl import AutoMLSelector
from models.timesnet import TimesNetStrategy
from models.autoformer import AutoformerStrategy
from models.fedformer import FEDformerStrategy
from backtest_engine import BacktestEngine

# ==========================================
# UI COMPONENTS
# ==========================================
def render_sidebar():
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", value="PTT.BK")
    start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", value=date.today())
    forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)
    
    st.sidebar.subheader("Model Settings")
    model_choice = st.sidebar.selectbox("Select Model", 
                                        ["RandomForest", "LSTM", "Prophet", 
                                         "XGBoost", "Hybrid (LSTM+XGB)", "AutoML",
                                         "TimesNet ⭐", "Autoformer ⭐", "FEDformer ⭐"])
    
    st.sidebar.subheader("Quant Features")
    use_quant = st.sidebar.checkbox("Enable Quant Features (Vol, Volatility)", value=True)
    
    st.sidebar.subheader("External Data")
    selected_factors = st.sidebar.multiselect("Global Markets", list(utils.GLOBAL_MARKETS.keys()))
    fred_key = st.sidebar.text_input("FRED API Key (Optional)", type="password")
    
    return {
        "ticker": ticker,
        "start": start_date,
        "end": end_date,
        "days": forecast_days,
        "model": model_choice,
        "quant": use_quant,
        "factors": selected_factors,
        "fred_key": fred_key
    }

def plot_charts(df):
    st.subheader("Market Analysis")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("Price & Trends", "Volume & OBV", "Volatility (ATR)"))
    
    # 1. Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
    if 'BB_Width' in df.columns:
        # Maybe plot Bands instead? For now BB Width is a feature.
        pass

    # 2. Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    if 'OBV' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='purple', width=1), name='OBV'), row=2, col=1)
        
    # 3. Volatility
    if 'ATR' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], line=dict(color='red', width=1), name='ATR'), row=3, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=800)
    st.plotly_chart(fig, use_container_width=True)

def display_insights(df, model_pred, trend):
    st.subheader("💡 Quant Insights")
    col1, col2, col3 = st.columns(3)
    
    # 1. Volatility Warning
    if 'ATR' in df.columns:
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].mean()
        if current_atr > avg_atr * 1.5:
            col1.error(f"High Volatility! ATR: {current_atr:.2f}")
        else:
            col1.success(f"Volatility Normal. ATR: {current_atr:.2f}")
            
    # 2. Trend Signal
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else 0
    price = df['Close'].iloc[-1]
    if price > sma_20:
        col2.metric("Trend (SMA20)", "Bullish", delta=f"{price-sma_20:.2f}")
    else:
        col2.metric("Trend (SMA20)", "Bearish", delta=f"{price-sma_20:.2f}")
        
    # 3. Model Sentiment
    sentiment = "Positive" if trend == "UP" else "Negative"
    col3.metric("Model Sentiment", sentiment)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    st.set_page_config(page_title="Quant Stock AI", layout="wide")
    st.title("🧠 Quant Stock Analysis & Forecasting")
    
    config = render_sidebar()
    
    # --- Data Loading ---
    with st.spinner("Loading Market Data..."):
        data = utils.load_main_ticker(config['ticker'], config['start'], config['end'])
        
        if data is None:
            st.error("Could not load ticker data.")
            return

        # External Data
        market_data = utils.load_global_markets(config['start'], config['end'])
        # Filter selected
        market_data = {k: v for k, v in market_data.items() if k in config['factors']}
        
        
        macro_data = utils.load_macro_data(config['fred_key'], config['start'], config['end'])
        
        # Display FRED Macro Data if available
        if macro_data:
            with st.expander("📊 ข้อมูลเศรษฐกิจมหภาค (FRED Data)", expanded=False):
                st.markdown("### ข้อมูลล่าสุดจาก Federal Reserve Economic Data")
                
                cols = st.columns(len(macro_data))
                for idx, (name, series) in enumerate(macro_data.items()):
                    with cols[idx]:
                        if not series.empty:
                            latest_value = series.iloc[-1]
                            prev_value = series.iloc[-2] if len(series) > 1 else latest_value
                            change = latest_value - prev_value
                            
                            st.metric(
                                label=name,
                                value=f"{latest_value:.2f}",
                                delta=f"{change:+.2f}"
                            )
                
                # Plot FRED data
                if len(macro_data) > 0:
                    fig_macro = go.Figure()
                    for name, series in macro_data.items():
                        if not series.empty:
                            fig_macro.add_trace(go.Scatter(
                                x=series.index,
                                y=series.values,
                                name=name,
                                mode='lines'
                            ))
                    
                    fig_macro.update_layout(
                        title="แนวโน้มข้อมูลเศรษฐกิจมหภาค",
                        xaxis_title="วันที่",
                        yaxis_title="ค่า",
                        height=400
                    )
                    st.plotly_chart(fig_macro, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("#### 💡 การตีความข้อมูล")
                    
                    for name, series in macro_data.items():
                        if not series.empty:
                            latest = series.iloc[-1]
                            
                            if "10Y Bond Yield" in name or "Bond" in name:
                                if latest > 4.5:
                                    st.warning(f"⚠️ **{name}**: {latest:.2f}% - สูง อาจกดดันตลาดหุ้น")
                                elif latest < 3.0:
                                    st.success(f"✅ **{name}**: {latest:.2f}% - ต่ำ เอื้อต่อตลาดหุ้น")
                                else:
                                    st.info(f"ℹ️ **{name}**: {latest:.2f}% - ปกติ")
                            
                            elif "CPI" in name:
                                # CPI is usually a large number (e.g., 300+), look at change
                                if len(series) > 12:
                                    yoy_change = (series.iloc[-1] / series.iloc[-13] - 1) * 100
                                    if yoy_change > 3:
                                        st.error(f"🔴 **{name}**: เงินเฟ้อสูง {yoy_change:.2f}% YoY")
                                    elif yoy_change < 2:
                                        st.success(f"✅ **{name}**: เงินเฟ้อต่ำ {yoy_change:.2f}% YoY")
                                    else:
                                        st.info(f"ℹ️ **{name}**: เงินเฟ้อปกติ {yoy_change:.2f}% YoY")
                            
                            elif "Fed Funds" in name or "Rate" in name:
                                if latest > 5:
                                    st.warning(f"⚠️ **{name}**: {latest:.2f}% - สูง ต้นทุนเงินทุนแพง")
                                elif latest < 2:
                                    st.success(f"✅ **{name}**: {latest:.2f}% - ต่ำ ต้นทุนเงินทุนถูก")
                                else:
                                    st.info(f"ℹ️ **{name}**: {latest:.2f}% - ปานกลาง")
        
        # Merge
        data = utils.merge_data(data, market_data, macro_data)
        
    # --- Feature Engineering ---
    if config['quant']:
        data = utils.add_quant_features(data)
    else:
        # Fallback to basic if needed, but utils.add_quant_features is robust
        data = utils.add_quant_features(data) 
        
    feature_cols = utils.prepare_features(data)
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis & Forecast", "Backtesting", "Model Insights", "📚 Glossary"])
    
    with tab1:
        plot_charts(data)
        
        st.subheader(f"Forecast ({config['model']})")
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, config['days'] + 1)]
        future_prices = []
        model = None
        
        with st.spinner(f"Training {config['model']}..."):
            if config['model'] == "RandomForest":
                strategy = RandomForestStrategy()
                model, scaler, _, _, _ = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data.iloc[-1], config['days'], feature_cols)
                
            elif config['model'] == "XGBoost":
                strategy = XGBoostStrategy()
                model, scaler, _, _, _ = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data.iloc[-1], config['days'], feature_cols)
                
            elif config['model'] == "Hybrid (LSTM+XGB)":
                strategy = HybridStrategy()
                # Hybrid returns tuple with multiple models
                lstm_m, xgb_m, scaler, _, _, seq_len = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(lstm_m, xgb_m, scaler, data, config['days'], seq_len, feature_cols)
                model = xgb_m # For SHAP later
                
            elif config['model'] == "AutoML":
                selector = AutoMLSelector()
                best_name, best_rmse, res = selector.find_best_model(data, feature_cols)
                st.success(f"AutoML selected: **{best_name}** (RMSE: {best_rmse:.4f})")
                
                # We need to forecast using the best model. 
                # This requires mapping back to the strategy.
                # Simplified: Re-instantiate strategy based on name
                if best_name == "RandomForest":
                    strat = RandomForestStrategy()
                    model, scaler, _, _, _ = res
                    future_prices = strat.forecast(model, scaler, data.iloc[-1], config['days'], feature_cols)
                elif best_name in ["XGBoost", "LightGBM"]:
                    strat = XGBoostStrategy()
                    model, scaler, _, _, _ = res
                    future_prices = strat.forecast(model, scaler, data.iloc[-1], config['days'], feature_cols)
            
            elif config['model'] == "LSTM":
                strategy = LSTMStrategy()
                model, scaler, _, _, seq_len, close_idx = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data, config['days'], seq_len, feature_cols, close_idx)

            elif config['model'] == "Prophet":
                strategy = ProphetStrategy()
                # Prophet handles features differently
                extra_features = [c for c in data.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
                model = strategy.train(data, extra_features)
                future_prices, future_dates, _ = strategy.forecast(model, data, config['days'], extra_features)
            
            elif config['model'] == "TimesNet ⭐":
                strategy = TimesNetStrategy()
                model, scaler, _, _, seq_len, feature_cols_used = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data, config['days'], seq_len, feature_cols)
            
            elif config['model'] == "Autoformer ⭐":
                strategy = AutoformerStrategy()
                model, scaler, _, _, seq_len, feature_cols_used = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data, config['days'], seq_len, feature_cols)
            
            elif config['model'] == "FEDformer ⭐":
                strategy = FEDformerStrategy()
                model, scaler, _, _, seq_len, feature_cols_used = strategy.train(data, feature_cols)
                future_prices = strategy.forecast(model, scaler, data, config['days'], seq_len, feature_cols)

        # Display Forecast
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices})
        if 'Date' in forecast_df.columns:
            forecast_df.set_index('Date', inplace=True)
        
        # Validate predictions (check for unrealistic values)
        last_p = data['Close'].iloc[-1]
        pred_values = np.array(future_prices)
        
        # Flag if predictions are unrealistic (>500% or <-90% change)
        max_change = np.max(np.abs((pred_values - last_p) / last_p * 100))
        if max_change > 500:
            st.warning(f"⚠️ โมเดลให้ผลลัพธ์ที่ผิดปกติ (เปลี่ยนแปลง {max_change:.0f}%) - แนะนำให้ลองโมเดลอื่น หรือตรวจสอบข้อมูล")
            # Clip extreme values for visualization
            pred_values = np.clip(pred_values, last_p * 0.5, last_p * 2.0)
            forecast_df['Predicted_Close'] = pred_values
            
        fig_f = go.Figure()
        
        # Historical data
        fig_f.add_trace(go.Scatter(
            x=data.index[-60:], 
            y=data['Close'][-60:], 
            name='ราคาจริง',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast
        fig_f.add_trace(go.Scatter(
            x=forecast_df.index, 
            y=forecast_df['Predicted_Close'], 
            name='พยากรณ์',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        # Update layout for better scaling
        fig_f.update_layout(
            title=f"การพยากรณ์ราคา {config['days']} วันข้างหน้า",
            xaxis_title="วันที่",
            yaxis_title="ราคา (บาท)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Auto-scale y-axis
        all_prices = list(data['Close'][-60:]) + list(forecast_df['Predicted_Close'])
        y_min = min(all_prices) * 0.95
        y_max = max(all_prices) * 1.05
        fig_f.update_yaxes(range=[y_min, y_max])
        
        st.plotly_chart(fig_f, use_container_width=True)
        
        # Comprehensive Prediction Summary
        st.subheader("📋 สรุปผลการพยากรณ์")
        
        last_p = data['Close'].iloc[-1]
        pred_p = future_prices[-1]
        trend = "UP" if pred_p > last_p else "DOWN"
        change_pct = ((pred_p - last_p) / last_p) * 100
        
        # Calculate prediction range
        pred_min = min(future_prices)
        pred_max = max(future_prices)
        pred_volatility = np.std(future_prices)
        
        # Summary Box
        st.info(f"""
        ### 🎯 สรุปหลัก
        
        **โมเดล:** {config['model']}  
        **ช่วงพยากรณ์:** {config['days']} วัน  
        **ราคาปัจจุบัน:** {last_p:.2f}  
        **ราคาพยากรณ์ (วันสุดท้าย):** {pred_p:.2f}  
        **การเปลี่ยนแปลง:** {change_pct:+.2f}%  
        **แนวโน้ม:** {"📈 ขาขึ้น (Bullish)" if trend == "UP" else "📉 ขาลง (Bearish)"}
        """)
        
        # Detailed Analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ราคาต่ำสุดที่คาด", f"{pred_min:.2f}", 
                     delta=f"{((pred_min - last_p) / last_p * 100):.2f}%")
        
        with col2:
            st.metric("ราคาสูงสุดที่คาด", f"{pred_max:.2f}",
                     delta=f"{((pred_max - last_p) / last_p * 100):.2f}%")
        
        with col3:
            st.metric("ความผันผวนที่คาด", f"{pred_volatility:.2f}")
        
        # Actionable Insights
        st.markdown("### 💡 คำแนะนำและข้อควรระวัง")
        
        # Trend Analysis
        if abs(change_pct) < 2:
            st.warning("⚠️ **แนวโน้มไม่ชัดเจน (Sideways)** - ราคาคาดว่าจะเคลื่อนไหวในกรอบแคบ แนะนำรอสัญญาณที่ชัดเจนกว่า")
        elif trend == "UP":
            if change_pct > 10:
                st.success(f"✅ **แนวโน้มขาขึ้นแรง** - โมเดลคาดการณ์ราคาขึ้น {change_pct:.2f}% ในช่วง {config['days']} วันข้างหน้า")
            else:
                st.success(f"✅ **แนวโน้มขาขึ้นปานกลาง** - โมเดลคาดการณ์ราคาขึ้น {change_pct:.2f}%")
        else:
            if change_pct < -10:
                st.error(f"🔴 **แนวโน้มขาลงแรง** - โมเดลคาดการณ์ราคาลง {abs(change_pct):.2f}% ควรระมัดระวัง")
            else:
                st.warning(f"⚠️ **แนวโน้มขาลงปานกลาง** - โมเดลคาดการณ์ราคาลง {abs(change_pct):.2f}%")
        
        # Volatility Warning
        if 'ATR' in data.columns:
            current_atr = data['ATR'].iloc[-1]
            avg_atr = data['ATR'].mean()
            if current_atr > avg_atr * 1.5:
                st.error("⚠️ **ความผันผวนสูง!** - ตลาดมีความผันผวนสูงกว่าปกติ ควรใช้ Stop Loss ที่กว้างขึ้น")
        
        # Technical Signals
        st.markdown("### 📊 สัญญาณทางเทคนิค")
        
        signals = []
        
        # RSI Signal
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                signals.append("🔴 RSI > 70: **Overbought** - ราคาอาจปรับตัวลง")
            elif rsi < 30:
                signals.append("🟢 RSI < 30: **Oversold** - ราคาอาจปรับตัวขึ้น")
            else:
                signals.append(f"⚪ RSI = {rsi:.1f}: อยู่ในกรอบปกติ")
        
        # MACD Signal
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            if macd > macd_signal:
                signals.append("🟢 MACD ตัดขึ้น: สัญญาณซื้อ")
            else:
                signals.append("🔴 MACD ตัดลง: สัญญาณขาย")
        
        # SMA Signal
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            price = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            sma50 = data['SMA_50'].iloc[-1]
            
            if price > sma20 and sma20 > sma50:
                signals.append("🟢 Golden Cross: ราคา > SMA20 > SMA50 (แนวโน้มขาขึ้น)")
            elif price < sma20 and sma20 < sma50:
                signals.append("🔴 Death Cross: ราคา < SMA20 < SMA50 (แนวโน้มขาลง)")
        
        for signal in signals:
            st.markdown(f"- {signal}")
        
        # Risk Management
        st.markdown("### 🛡️ การบริหารความเสี่ยง")
        
        # Calculate suggested stop loss and take profit
        risk_pct = 3  # 3% risk
        reward_pct = 6  # 6% reward (Risk:Reward = 1:2)
        
        if trend == "UP":
            entry = last_p
            stop_loss = entry * (1 - risk_pct/100)
            take_profit = entry * (1 + reward_pct/100)
            
            st.success(f"""
            **สำหรับการซื้อ (Long Position):**
            - 📍 จุดเข้า: {entry:.2f}
            - 🛑 Stop Loss: {stop_loss:.2f} (-{risk_pct}%)
            - 🎯 Take Profit: {take_profit:.2f} (+{reward_pct}%)
            - 📊 Risk:Reward = 1:2
            """)
        else:
            entry = last_p
            stop_loss = entry * (1 + risk_pct/100)
            take_profit = entry * (1 - reward_pct/100)
            
            st.warning(f"""
            **สำหรับการขาย (Short Position):**
            - 📍 จุดเข้า: {entry:.2f}
            - 🛑 Stop Loss: {stop_loss:.2f} (+{risk_pct}%)
            - 🎯 Take Profit: {take_profit:.2f} (-{reward_pct}%)
            - 📊 Risk:Reward = 1:2
            """)
        
        # Disclaimer
        st.error("""
        ⚠️ **ข้อจำกัดความรับผิดชอบ**
        
        การพยากรณ์นี้เป็นเพียงการวิเคราะห์ทางสถิติและ AI เท่านั้น ไม่ใช่คำแนะนำในการลงทุน 
        ตลาดหุ้นมีความเสี่ยงสูงและผันผวน ผลการพยากรณ์อาจไม่ตรงกับความเป็นจริง 
        กรุณาศึกษาข้อมูลเพิ่มเติมและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน
        """)
        
        display_insights(data, future_prices, trend)

    with tab2:
        st.subheader("Walk-Forward Backtesting")
        if st.button("Run Walk-Forward Test"):
            # Simple wrapper for backtest
            # Note: Only works easily for standard models (RF, XGB) in this demo structure
            if config['model'] in ["RandomForest", "XGBoost"]:
                if config['model'] == "RandomForest":
                    strat = RandomForestStrategy()
                else:
                    strat = XGBoostStrategy()
                    
                engine = BacktestEngine(strat)
                with st.spinner("Running Walk-Forward Validation..."):
                    res = engine.walk_forward_validation(data, feature_cols)
                    st.dataframe(res)
                    st.metric("Average RMSE", f"{res['RMSE'].mean():.4f}")
            else:
                st.warning("Walk-Forward Backtest currently supported for RandomForest and XGBoost only.")

    with tab3:
        st.subheader("Explainable AI (SHAP)")
        if config['model'] in ["RandomForest", "XGBoost", "AutoML"] and model is not None:
            if st.button("Calculate SHAP Values"):
                with st.spinner("Calculating SHAP..."):
                    # Need X_train/test for SHAP. 
                    # We don't have easy access here without re-splitting.
                    # Let's just take a sample of data
                    X = data[feature_cols]
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X.iloc[-100:]) # Last 100 days
                    
                    fig, ax = plt.subplots()
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(fig, clear_figure=True)
        else:
            st.info("SHAP explanation available for Tree-based models (RF, XGBoost).")

    with tab4:
        st.subheader("📚 คำศัพท์เฉพาะทาง (Technical Glossary)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 ตัวชี้วัดทางเทคนิค (Technical Indicators)")
            
            with st.expander("**SMA** - Simple Moving Average"):
                st.write("""
                **ค่าเฉลี่ยเคลื่อนที่แบบธรรมดา**
                - คำนวณจากราคาปิดเฉลี่ยย้อนหลัง N วัน
                - SMA 20 = ค่าเฉลี่ย 20 วัน, SMA 50 = ค่าเฉลี่ย 50 วัน
                - ใช้ดูแนวโน้มราคาระยะสั้น-กลาง
                """)
            
            with st.expander("**EMA** - Exponential Moving Average"):
                st.write("""
                **ค่าเฉลี่ยเคลื่อนที่แบบเอ็กซ์โพเนนเชียล**
                - ให้น้ำหนักกับราคาล่าสุดมากกว่า SMA
                - ตอบสนองต่อการเปลี่ยนแปลงราคาเร็วกว่า
                - เหมาะสำหรับการเทรดระยะสั้น
                """)
            
            with st.expander("**RSI** - Relative Strength Index"):
                st.write("""
                **ดัชนีความแข็งแกร่งสัมพัทธ์**
                - วัดความเร็วและการเปลี่ยนแปลงของราคา
                - ค่า 0-100: >70 = Overbought, <30 = Oversold
                - ช่วยบ่งชี้จุดกลับตัวของราคา
                """)
            
            with st.expander("**MACD** - Moving Average Convergence Divergence"):
                st.write("""
                **ตัวชี้วัดการลู่เข้าและแยกออกของค่าเฉลี่ย**
                - MACD Line = EMA(12) - EMA(26)
                - Signal Line = EMA(9) ของ MACD
                - ตัดกันขึ้น = สัญญาณซื้อ, ตัดกันลง = สัญญาณขาย
                """)
            
            with st.expander("**ATR** - Average True Range"):
                st.write("""
                **ค่าเฉลี่ยช่วงความผันผวนที่แท้จริง**
                - วัดความผันผวนของราคา
                - ATR สูง = ตลาดผันผวนมาก
                - ใช้กำหนด Stop Loss และ Take Profit
                """)
            
            with st.expander("**Bollinger Bands**"):
                st.write("""
                **แถบโบลลิงเจอร์**
                - แถบบน/ล่าง = SMA ± (2 × ส่วนเบี่ยงเบนมาตรฐาน)
                - BB Width = ระยะห่างระหว่างแถบ
                - แถบแคบ = ตลาดสงบ, แถบกว้าง = ตลาดผันผวน
                """)
        
        with col2:
            st.markdown("### 📈 ตัวชี้วัดปริมาณ (Volume Indicators)")
            
            with st.expander("**OBV** - On-Balance Volume"):
                st.write("""
                **ปริมาณการซื้อขายสะสม**
                - รวมปริมาณเมื่อราคาขึ้น, ลบเมื่อราคาลง
                - OBV ขึ้น = แรงซื้อเพิ่ม
                - ใช้ยืนยันแนวโน้มราคา
                """)
            
            with st.expander("**MFI** - Money Flow Index"):
                st.write("""
                **ดัชนีกระแสเงิน**
                - RSI แบบมีปริมาณการซื้อขาย
                - >80 = Overbought, <20 = Oversold
                - วัดแรงซื้อ-ขายที่แท้จริง
                """)
            
            st.markdown("### 🤖 โมเดล AI (AI Models)")
            
            with st.expander("**RandomForest**"):
                st.write("""
                **ป่าสุ่ม**
                - รวมผลจาก Decision Trees หลายต้น
                - แม่นและไม่ Overfit ง่าย
                - เหมาะกับข้อมูลที่มีหลาย Features
                """)
            
            with st.expander("**XGBoost**"):
                st.write("""
                **Extreme Gradient Boosting**
                - อัลกอริทึม Boosting ที่เร็วและแม่น
                - เหมาะกับข้อมูลตาราง (Tabular)
                - ใช้ใน Kaggle และ Quant Trading
                """)
            
            with st.expander("**LSTM** - Long Short-Term Memory"):
                st.write("""
                **หน่วยความจำระยะสั้น-ยาว**
                - โครงข่ายประสาทเทียมแบบ Recurrent
                - จับ Pattern ข้ามเวลาได้ดี
                - เหมาะกับข้อมูล Time Series
                """)
            
            with st.expander("**Prophet**"):
                st.write("""
                **โมเดลพยากรณ์จาก Facebook**
                - แยกแนวโน้ม, ฤดูกาล, วันหยุด
                - ใช้งานง่าย, ไม่ต้องปรับแต่งมาก
                - เหมาะกับข้อมูลที่มี Seasonality
                """)
            
            with st.expander("**Hybrid (LSTM+XGB)**"):
                st.write("""
                **โมเดลผสม**
                - LSTM จับ Pattern ระยะสั้น
                - XGBoost ใช้ Technical Features
                - รวมจุดแข็งของทั้งสองโมเดล
                """)
            
            with st.expander("**TimesNet ⭐** - NEW"):
                st.write("""
                **โมเดล State-of-the-Art 2023**
                - แปลง Time Series เป็น 2D Image
                - ใช้ CNN จับ Temporal Patterns
                - แม่นสูงสำหรับข้อมูลที่มี Periodicity
                - เหมาะกับหุ้นที่มีรูปแบบซ้ำ
                """)
            
            with st.expander("**Autoformer ⭐** - NEW"):
                st.write("""
                **Auto-Correlation Transformer**
                - ใช้ Auto-Correlation แทน Self-Attention
                - แยก Trend และ Seasonal Components
                - เร็วกว่า Transformer ปกติ
                - เหมาะกับข้อมูลที่มี Seasonality ชัดเจน
                """)
            
            with st.expander("**FEDformer ⭐** - NEW"):
                st.write("""
                **Frequency Enhanced Decomposed Transformer**
                - ทำงานใน Frequency Domain (FFT)
                - จับ Pattern ระยะยาวได้ดีมาก
                - ประหยัดหน่วยความจำ
                - เหมาะกับการพยากรณ์ระยะยาว
                """)

        
        st.markdown("---")
        st.markdown("### 📉 เมตริกประเมินโมเดล (Model Metrics)")
        
        col3, col4 = st.columns(2)
        
        with col3:
            with st.expander("**MAE** - Mean Absolute Error"):
                st.write("""
                **ค่าเฉลี่ยความผิดพลาดสัมบูรณ์**
                - ค่าเฉลี่ยของ |ราคาจริง - ราคาพยากรณ์|
                - ยิ่งต่ำยิ่งดี
                - หน่วยเดียวกับราคา (บาท/ดอลลาร์)
                """)
        
        with col4:
            with st.expander("**RMSE** - Root Mean Squared Error"):
                st.write("""
                **รากที่สองของค่าเฉลี่ยความผิดพลาดกำลังสอง**
                - ลงโทษความผิดพลาดใหญ่มากกว่า MAE
                - ยิ่งต่ำยิ่งดี
                - ใช้เปรียบเทียบโมเดล
                """)
        
        with col3:
            with st.expander("**Sharpe Ratio**"):
                st.write("""
                **อัตราส่วนชาร์ป**
                - วัดผลตอบแทนต่อความเสี่ยง
                - (Return - Risk-free Rate) / Std Dev
                - >1 = ดี, >2 = ดีมาก, >3 = ยอดเยี่ยม
                """)
        
        with col4:
            with st.expander("**Max Drawdown**"):
                st.write("""
                **การขาดทุนสูงสุด**
                - % ที่ลดลงจากจุดสูงสุด
                - วัดความเสี่ยงที่เลวร้ายที่สุด
                - ยิ่งต่ำยิ่งดี (เช่น -10% ดีกว่า -30%)
                """)
        
        st.markdown("---")
        st.markdown("### 🌍 ข้อมูลภายนอก (External Data)")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("#### 📊 ตลาดโลก (Global Markets)")
            
            with st.expander("**VIX** - Volatility Index"):
                st.write("""
                **ดัชนีความผันผวน (Fear Index)**
                - วัดความกลัวในตลาดหุ้นสหรัฐ
                - VIX < 15 = ตลาดสงบ
                - VIX 15-25 = ปกติ
                - VIX > 25 = ตลาดผันผวนสูง
                - มักเคลื่อนไหวตรงข้ามกับตลาดหุ้น
                """)
            
            with st.expander("**S&P 500 Futures (ES=F)**"):
                st.write("""
                **สัญญาซื้อขายล่วงหน้าดัชนี S&P 500**
                - สะท้อนทิศทางตลาดหุ้นสหรัฐ
                - ซื้อขายได้ 24 ชั่วโมง
                - มีผลต่อตลาดหุ้นทั่วโลก
                - ใช้ดูแนวโน้มก่อนตลาดเปิด
                """)
            
            with st.expander("**Nasdaq Futures (NQ=F)**"):
                st.write("""
                **สัญญาซื้อขายล่วงหน้าดัชนี Nasdaq**
                - เน้นหุ้นเทคโนโลยี
                - มีความผันผวนสูงกว่า S&P 500
                - มีผลต่อหุ้นเทคโนโลยีทั่วโลก
                """)
            
            with st.expander("**Dollar Index (DX-Y.NYB)**"):
                st.write("""
                **ดัชนีค่าเงินดอลลาร์**
                - วัดความแข็งแกร่งของดอลลาร์
                - ดอลลาร์แข็ง = ทองคำ/น้ำมันอ่อน
                - มีผลต่อการส่งออกและนำเข้า
                - ดัชนีขึ้น = ดอลลาร์แข็งค่า
                """)
        
        with col6:
            st.markdown("#### 🛢️ สินค้าโภคภัณฑ์ (Commodities)")
            
            with st.expander("**Gold (GC=F)**"):
                st.write("""
                **ทองคำ**
                - สินทรัพย์ปลอดภัย (Safe Haven)
                - ราคาขึ้นเมื่อตลาดหุ้นลง
                - ป้องกันเงินเฟ้อ
                - เคลื่อนไหวตรงข้ามดอลลาร์
                """)
            
            with st.expander("**Crude Oil (CL=F)**"):
                st.write("""
                **น้ำมันดิบ**
                - สินค้าโภคภัณฑ์สำคัญ
                - มีผลต่อต้นทุนการผลิต
                - ราคาขึ้น = เงินเฟ้อเพิ่ม
                - สำคัญต่อหุ้นพลังงาน (เช่น PTT)
                """)
            
            with st.expander("**SET Index (^SET.BK)**"):
                st.write("""
                **ดัชนีตลาดหลักทรัพย์แห่งประเทศไทย**
                - รวมหุ้นใหญ่ในตลาดไทย
                - สะท้อนเศรษฐกิจไทย
                - มีผลต่อหุ้นไทยทั้งหมด
                """)
            
            st.markdown("#### 📈 ข้อมูลเศรษฐกิจมหภาค (FRED API)")
            
            with st.expander("**10Y Bond Yield**"):
                st.write("""
                **อัตราผลตอบแทนพันธบัตร 10 ปี**
                - สะท้อนความเชื่อมั่นเศรษฐกิจ
                - ขึ้น = เงินไหลออกจากหุ้น
                - ลง = เงินไหลเข้าหุ้น
                """)
            
            with st.expander("**CPI & Fed Funds Rate**"):
                st.write("""
                **ดัชนีราคาผู้บริโภค & อัตราดอกเบี้ย**
                - CPI = วัดเงินเฟ้อ
                - Fed Rate = อัตราดอกเบี้ยนโยบาย
                - มีผลต่อต้นทุนเงินทุน
                """)



if __name__ == "__main__":
    main()
