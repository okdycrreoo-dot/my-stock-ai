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

# --- 1. 配置與專業深色主題 ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00F5FF; font-weight: bold; }
    .stMetric { background-color: #1C2128; border: 2px solid #30363D; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎：技術指標計算 ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    try:
        data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 趨勢指標
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        std = data['Close'].rolling(20).std()
        data['BB_up'] = data['MA20'] + (std * 2)
        data['BB_low'] = data['MA20'] - (std * 2)
        
        # MACD 計算
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Hist'] = data['MACD'] - data['Signal']
        
        # 支撐與壓力
        recent = data.tail(60)
        data['Support'] = recent['Low'].min()
        data['Resistance'] = recent['High'].max()
        
        return data.dropna()
    except: return None

# --- 3. 視覺強化繪圖引擎 ---
def show_ultimate_dashboard(symbol, unit, p_days, precision):
    df = fetch_comprehensive_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取股票代碼 '{symbol}'")
        return

    # AI 向量化預測
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    trend = (int(precision) - 55) / 500
    pred_prices = last_p * np.cumprod(1 + trend + noise)

    # 儀表板卡片
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
    c3.metric("預期回報", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # 繪製圖表 (視覺強化版)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.55, 0.15, 0.3], vertical_spacing=0.04)
    
    zoom = {"日": 45, "月": 180, "年": 500}[unit]
    p_df = df.tail(zoom)
    
    # --- 第一層：主圖 (強化 K 線與 MA) ---
    fig.add_trace(go.Candlestick(
        x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], 
        name='K線', increasing_line_color='#00FF41', decreasing_line_color='#FF3131',
        increasing_fillcolor='#00FF41', decreasing_fillcolor='#FF3131'
    ), row=1, col=1)

    # 均線加粗：width=2.5
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5 (週)', line=dict(color='#FFFF00', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20 (月)', line=dict(color='#00F5FF', width=2.5)), row=1, col=1)
    
    # 布林通道改為虛線，避免干擾
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_up'], name='布林上', line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_low'], name='布林下', fill='tonexty', fillcolor='rgba(255,255,255,0.05)', line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')), row=1, col=1)
    
    # 支撐壓力加粗
    fig.add_hline(y=p_df['Support'].iloc[-1], line_dash="dash", line_color="#00FF41", line_width=2, row=1, col=1)
    fig.add_hline(y=p_df['Resistance'].iloc[-1], line_dash="dash", line_color="#FF3131", line_width=2, row=1, col=1)
    
    # AI 預測路徑強化：火焰橘 dashdot
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4, dash='dashdot')), row=1, col=1)

    # --- 第二層：成交量 (增加對比度) ---
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors, opacity=0.7), row=2, col=1)

    # --- 第三層：MACD (高飽和度線條) ---
    # MACD 加粗
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2)), row=3, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['Signal'], name='Signal', line=dict(color='#FFD700', width=2)), row=3, col=1)
    
    # MACD 柱狀圖顏色調整
    hist_colors = ['#FF3131' if val < 0 else '#00FF41' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='Hist', marker_color=hist_colors, opacity=0.4), row=3, col=1)

    # 佈局微調
    fig.update_layout(
        template="plotly_dark", 
        height=900, 
        xaxis_rangeslider_visible=False, 
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # 增加網格線亮度
    fig.update_xaxes(gridcolor='#333333', zeroline=False)
    fig.update_yaxes(gridcolor='#333333', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 4. 主程式 (與原邏輯一致，僅修正 UI 連結) ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
    except:
        st.error("🚨 系統連線異常，請檢查 Secrets。")
        return

    try:
        ws_settings = sh.worksheet("settings")
        s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
        curr_prec = int(s_data.get('global_precision', 55))
        curr_ttl = int(s_data.get('api_ttl_min', 5))
    except:
        curr_prec, curr_ttl = 55, 5

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級技術終端")
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        if st.button("進入系統"):
            user_df = pd.DataFrame(sh.worksheet("users").get_all_records())
            if not user_df[(user_df['username'].astype(str)==u) & (user_df['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
    else:
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.caption(f"👤 {st.session_state.user} | ⏳ 刷新倒數: {max(0, int(remain.total_seconds()))}s")

        with st.sidebar:
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員控制台", expanded=True):
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("快取分鐘", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("同步所有設定"):
                        ws_settings.update_cell(2, 2, str(new_p))
                        ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear()
                        st.session_state.last_sync = datetime.now(); st.rerun()
            
            ws_watch = sh.worksheet("watchlist")
            all_w = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
            
            target = st.selectbox("我的清單", user_stocks if user_stocks else ["2330.TW"])
            unit = st.selectbox("圖表單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測天數", 1, 30, 7)
            
            new_s = st.text_input("新增代碼 (如: AAPL)").strip().upper()
            if st.button("確認新增"):
                if new_s and new_s not in user_stocks:
                    ws_watch.append_row([st.session_state.user, new_s]); st.rerun()
            
            if st.button("🚪 登出"):
                st.session_state.user = None; st.rerun()

        show_ultimate_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
