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

# --- 1. 核心配置與視覺樣式 ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")
ICON_URL = "https://raw.githubusercontent.com/okdycrreoo-dot/kdstockai/main/icon.png"

# PWA 與樣式注入 (簡化合併版)
st.markdown(f"""
    <link rel="manifest" href="./manifest.json?v=7.0">
    <link rel="apple-touch-icon" href="{ICON_URL}?v=7.0"><link rel="icon" type="image/png" href="{ICON_URL}?v=7.0">
    <style>
    .stApp {{ background-color: #0E1117; color: #FFFFFF !important; }}
    label, p, span, .stMarkdown, .stCaption {{ color: #FFFFFF !important; font-weight: 600 !important; }}
    .streamlit-expanderHeader {{ background-color: #1C2128 !important; color: #00F5FF !important; border: 2px solid #00F5FF !important; border-radius: 12px !important; font-weight: 800 !important; }}
    div[data-baseweb="select"] > div {{ background-color: #1C2128 !important; color: #FFFFFF !important; border: 2px solid #00F5FF !important; }}
    div[role="listbox"] div {{ color: #000000 !important; }} 
    .stButton>button {{ background-color: #00F5FF !important; color: #0E1117 !important; border-radius: 12px; font-weight: 900 !important; height: 3.2rem !important; }}
    .diag-box {{ background-color: #161B22; border-left: 5px solid #00F5FF; border-radius: 10px; padding: 15px; margin-bottom: 10px; border: 1px solid #30363D; }}
    .price-tag {{ font-size: 1.1rem; font-weight: 800; }}
    button[data-testid="sidebar-button"] {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎：日線為基底 (KDJ/MACD/Volume) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for _ in range(3):
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA60'] = data['Close'].rolling(60).mean()
                # MACD
                exp1, exp2 = data['Close'].ewm(span=12).mean(), data['Close'].ewm(span=26).mean()
                data['Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
                # KDJ
                low, high = data['Low'].rolling(9).min(), data['High'].rolling(9).max()
                rsv = (data['Close'] - low) / (high - low) * 100
                data['J'] = 3 * rsv.ewm(com=2).mean() - 2 * rsv.ewm(com=2).mean().ewm(com=2).mean()
                data['V_MA5'] = data['Volume'].rolling(5).mean()
                return data.dropna()
            time.sleep(1)
        except: time.sleep(1); continue
    return None

# --- 3. AI 買賣建議價分析 (多指標卷積) ---
def perform_ai_advice(df, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    bias = (int(precision) - 55) / 100
    def get_p(ma, cycle):
        v = 0.03 if cycle=="5D" else (0.06 if cycle=="20D" else 0.10)
        mod = (0.01 if last['Hist'] > prev['Hist'] else -0.01) + (0.01 if last['Volume'] > last['V_MA5'] else 0)
        return ma * (1 - v + mod + bias), ma * (1 + v + mod + bias)
    return {"5日(短線)": get_p(last['MA5'], "5D"), "20日(月線)": get_p(last['MA20'], "20D"), "60日(波段)": get_p(last['MA60'], "60D")}

# --- 4. 儀表板繪製 ---
def show_dashboard(symbol, p_days, precision):
    df = fetch_comprehensive_data(symbol)
    if df is None: st.error("數據獲取失敗"); return
    
    last_p = float(df['Close'].iloc[-1])
    advice = perform_ai_advice(df, precision)
    pred = last_p * np.cumprod(1 + (int(precision)-55)/500 + np.random.normal(0, 0.002, p_days))

    # 指標卡
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價", f"{last_p:.2f}")
    c2.metric(f"AI預估({p_days}天)", f"{pred[-1]:.2f}")
    c3.metric("預期回報", f"{((pred[-1]-last_p)/last_p)*100:.2f}%")

    # 買賣建議盒
    st.markdown("### 🤖 AI 智能買賣點建議 (MACD/KDJ/量)")
    cols = st.columns(3)
    for i, (k, v) in enumerate(advice.items()):
        with cols[i]:
            st.markdown(f"<div class='diag-box'><b>{k}</b><br>🟢 買入: <span class='price-tag' style='color:#00FF41'>{v[0]:.2f}</span><br>🔴 賣出: <span class='price-tag' style='color:#FF3131'>{v[1]:.2f}</span></div>", unsafe_allow_html=True)

    # 圖表
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    p_df = df.tail(60)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF')), 1, 1)
    fig.add_trace(go.Scatter(x=[p_df.index[-1] + timedelta(days=i) for i in range(1, p_days+1)], y=pred, name='AI預測', line=dict(color='#FF4500', dash='dashdot')), 1, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color='#30363D'), 2, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='KDJ-J', line=dict(color='#FF00FF')), 3, 1)
    fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --- 5. 主程式 (1小時登入) ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    @st.cache_resource(ttl=3600)
    def auth_v(): return {"u": None}
    vault = auth_v()
    if not st.session_state.user: st.session_state.user = vault["u"]

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: st.error("數據庫連線異常"); return

    sd = {i['setting_name']: i['value'] for i in ws_s.get_all_records()}
    cp, ct = int(sd.get('global_precision', 55)), int(sd.get('api_ttl_min', 5))

    if not st.session_state.user:
        st.title("🚀 StockAI 登入")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("確認登入", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = vault["u"] = u; st.rerun()
            else: st.error("帳密錯誤")
    else:
        with st.expander("⚙️ 功能面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_s = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("選股", user_s if user_s else ["2330.TW"])
                ns = st.text_input("➕ 新增代碼")
                if st.button("確認新增") and ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                # 核心修正：取消單位，標註最大30日
                p_days = st.number_input("AI 預測天數 (最大 30 日)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    npw = st.slider("全域靈敏度", 0, 100, cp)
                    nt = st.select_slider("快取連線時間 (分)", options=list(range(1, 11)), value=ct)
                    if st.button("💾 同步資料庫"):
                        ws_s.update_cell(2, 2, str(npw)); ws_s.update_cell(3, 2, str(nt)); st.rerun()
                if st.button("🚪 登出"): st.session_state.user = vault["u"] = None; st.rerun()
        show_dashboard(target, p_days, cp)

if __name__ == "__main__": main()
