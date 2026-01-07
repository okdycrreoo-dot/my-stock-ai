import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. 頁面配置與高對比風格 ---
st.set_page_config(page_title="StockAI 高級技術終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據抓取與技術指標計算 ---
@st.cache_data(ttl=600)
def fetch_and_process_data(symbol):
    try:
        data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # --- 專業技術指標計算 ---
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # 布林通道 (20日, 2標準差)
        std = data['Close'].rolling(window=20).std()
        data['BB_up'] = data['MA20'] + (std * 2)
        data['BB_low'] = data['MA20'] - (std * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        return data.dropna()
    except: return None

# --- 3. 繪圖引擎 (整合所有專業線條) ---
def show_analysis_dashboard(symbol, unit, p_days, precision):
    df = fetch_and_process_data(symbol)
    if df is None:
        st.error(f"❌ 無法解析 {symbol} 數據。")
        return

    # AI 預測邏輯 (保留前述專業權重)
    last_p = float(df['Close'].iloc[-1])
    rsi_last = float(df['RSI'].iloc[-1])
    ma20_last = float(df['MA20'].iloc[-1])
    
    # 預測運算
    pred_prices = []
    curr_p = last_p
    for i in range(1, p_days + 1):
        bias = ((50 - rsi_last) * 0.001) + ((ma20_last - curr_p) / ma20_last * 0.02)
        curr_p *= (1 + (int(precision)/100 * 0.01) + bias + np.random.normal(0, 0.0015))
        pred_prices.append(float(curr_p))

    # --- 介面卡片 ---
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("現價", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}d)", f"{target_p:.2f}")
    c3.metric("預期回報", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # --- 繪製專業複合圖表 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    
    zoom = {"日": 45, "月": 200, "年": 600}[unit]
    p_df = df.tail(zoom)
    
    # A. 主圖指標 (K線 + 均線 + 布林)
    # 1. K線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線', opacity=0.8), row=1, col=1)
    # 2. 收盤價線 (白色細線)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Close'], name='收盤線', line=dict(color='rgba(255,255,255,0.5)', width=1)), row=1, col=1)
    # 3. 均線 MA5 (黃色), MA20 (青色)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5 (週)', line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20 (月)', line=dict(color='#00F5FF', width=1.5)), row=1, col=1)
    # 4. 布林通道 (半透明灰色區間)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BB_up'], name='布林上軌', line=dict(color='#444', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BB_low'], name='布林下軌', line=dict(color='#444', dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # B. AI 預測延伸
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_prices, name='AI 預測路徑', line=dict(color='#FF4500', width=3, dash='dashdot')), row=1, col=1)

    # C. 交易量 (紅跌綠漲)
    colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='交易量', marker_color=colors, opacity=0.7), row=2, col=1)

    # 圖表格式優化
    fig.update_layout(template="plotly_dark", height=750, paper_bgcolor="#0E1117", plot_bgcolor="#111418", 
                      xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#222')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#222')
    
    st.plotly_chart(fig, use_container_width=True)

# (其餘 main 邏輯保持不變...)
