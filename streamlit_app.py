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

# --- 1. 頁面配置與快取優化 ---
st.set_page_config(page_title="StockAI 高速終端", layout="wide")

# 減少渲染負擔的 CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .stPlotlyChart { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 高效數據引擎 ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_fast_data(symbol):
    try:
        # 僅抓取必要長度，減少 I/O 時間
        data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 向量化計算指標 (效率最高)
        close = data['Close']
        data['MA5'] = close.rolling(5).mean()
        data['MA20'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        data['BB_up'] = data['MA20'] + (std * 2)
        data['BB_low'] = data['MA20'] - (std * 2)
        
        # 支撐壓力位 (最近 60 天極值)
        recent = data.tail(60)
        data['Support'] = recent['Low'].min()
        data['Resistance'] = recent['High'].max()
        
        return data.dropna()
    except: return None

# --- 3. 繪圖與 AI 引擎 (GL 加速版) ---
def show_analysis_dashboard(symbol, unit, p_days, precision):
    df = fetch_fast_data(symbol)
    if df is None:
        st.error("數據獲取失敗")
        return

    # AI 預測 (向量化模擬)
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    # 建立趨勢權重
    trend = (int(precision)-55) / 500
    pred_ratios = np.cumprod(1 + trend + noise)
    pred_prices = last_p * pred_ratios

    # 圖表對比優化
    zoom = {"日": 40, "月": 180, "年": 500}[unit]
    p_df = df.tail(zoom)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)

    # 1. K 線 (主軸)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    
    # 2. 專業線條 (使用 Scattergl 提升渲染速度)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), row=1, col=1)
    
    # 3. 支撐與壓力 (橫向虛線)
    fig.add_hline(y=p_df['Support'].iloc[-1], line_dash="dash", line_color="green", annotation_text="支撐", row=1, col=1)
    fig.add_hline(y=p_df['Resistance'].iloc[-1], line_dash="dash", line_color="red", annotation_text="壓力", row=1, col=1)

    # 4. 布林通道 (半透明填充)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_up'], name='BB上軌', line=dict(width=0)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_low'], name='BB下軌', line=dict(width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.15)'), row=1, col=1)

    # 5. AI 預測
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=3, dash='dashdot')), row=1, col=1)

    # 6. 成交量
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color='#30363D'), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# (其餘管理員設定與 main 邏輯同前，確保 okdycrreoo 擁有 TTL 控制權)
