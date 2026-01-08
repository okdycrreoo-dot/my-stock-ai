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

# --- 1. 配置與 UI 深度修復 (解決面板文字隱沒問題) ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")
ICON_URL = "https://raw.githubusercontent.com/okdycrreoo-dot/my-stock-ai/main/icon.png"

st.markdown(f"""
    <link rel="icon" type="image/png" href="{ICON_URL}">
    <style>
    /* 1. 背景與基礎文字 */
    .stApp {{ background-color: #0E1117; color: #FFFFFF !important; }}
    
    /* 2. 控制面板標籤文字強制亮白 (關鍵修正) */
    .stMarkdown, p, label, .stSlider label, .stNumberInput label, .stSelectbox label {{ 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px #000000;
        margin-bottom: 5px !important;
    }}
    
    /* 3. 修正 Slider 數值文字 */
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], div[data-baseweb="slider"] {{
        color: #00F5FF !important;
    }}

    /* 4. 按鈕視覺：純白文字配青色背景 */
    .stButton>button {{ 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border-radius: 10px; 
        font-weight: 900 !important; 
        height: 3.5rem;
        border: 2px solid #FFFFFF;
    }}
    
    /* 5. 區塊樣式 */
    .diag-box {{ background-color: #161B22; border-left: 5px solid #00F5FF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #30363D; }}
    .summary-card {{ background-color: #1C2128; border-radius: 10px; padding: 20px; border: 1px solid #00F5FF; border-left: 8px solid #FF4500; }}
    .price-tag {{ font-size: 1.3rem; font-weight: 900; }}
    
    /* 6. 隱藏側欄 */
    button[data-testid="sidebar-button"] {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據與 AI 摘要引擎 ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_and_summary(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="2y", interval="1d", auto_adjust=True)
        if data.empty: return None, "無數據"
        
        # 指標計算 (保持穩定性)
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA60'] = data['Close'].rolling(60).mean()
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Hist'] = data['MACD'] - data['Signal']
        low_9, high_9 = data['Low'].rolling(9).min(), data['High'].rolling(9).max()
        rsv = (data['Close'] - low_9) / (high_9 - low_9) * 100
        data['K'] = rsv.ewm(com=2, adjust=False).mean()
        data['D'] = data['K'].ewm(com=2, adjust=False).mean()
        data['J'] = 3 * data['K'] - 2 * data['D']
        
        # AI 新聞摘要處理
        search = yf.Search(symbol, max_results=5)
        news_titles = [n.get('title', '') for n in search.news]
        if news_titles:
            summary = " | ".join(news_titles[:3]) # 擷取前三則作為重點摘要
        else:
            summary = "目前市場無重大更新。"
            
        return data.dropna(), summary
    except: return None, "獲取摘要失敗"

# --- 3. 儀表板繪製 ---
def show_dashboard(symbol, p_days, precision):
    df, news_summary = fetch_data_and_summary(symbol)
    if df is None: st.error("⚠️ 無法載入數據，請重新確認代碼。"); return
    
    # A. 買賣建議區
    bias = (int(precision) - 55) / 100
    st.markdown("### 🤖 AI 智能交易訊號")
    c1, c2, c3 = st.columns(3)
    p_pts = {"5日(短)": 0.03, "20日(中)": 0.06, "60日(長)": 0.10}
    for i, (k, v) in enumerate(p_pts.items()):
        ma_val = df[f'MA{k[0:k.find("(")]}'].iloc[-1]
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='diag-box'><b>{k}</b><br>🟢 買點: <span class='price-tag' style='color:#00FF41'>{(ma_val*(1-v+bias)):.2f}</span><br>🔴 賣點: <span class='price-tag' style='color:#FF3131'>{(ma_val*(1+v+bias)):.2f}</span></div>", unsafe_allow_html=True)

    # B. 四層加粗圖表 (視覺比照 image_3f9201.png)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
    p_df = df.tail(60)
    # K線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    # 成交量
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]), 2, 1)
    # MACD 青色加粗
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD線', line=dict(color='#00F5FF', width=3)), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], name='訊號線', line=dict(color='#FFFF00', width=1.5)), 3, 1)
    # KDJ 綠色加粗 (K線)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(綠)', line=dict(color='#00FF41', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J(紫)', line=dict(color='#FF00FF', width=1.5)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)

    # C. AI 新聞重點說明 (取代繁雜清單)
    st.markdown("### 📰 AI 市場重點解析")
    st.markdown(f"""
        <div class='summary-card'>
            <h4 style='color:#FF4500; margin-top:0;'>本日核心新聞摘要：</h4>
            <p style='color:#FFFFFF; font-size:1.1rem; line-height:1.6;'>{news_summary}</p>
            <hr style='border: 0.5px solid #30363D;'>
            <p style='color:#00F5FF;'><b>💡 AI 建議：</b> 目前市場訊息頻繁，建議觀察 {symbol} 在 MA20 支撐位階，配合 MACD 動能判斷進場時機。</p>
        </div>
    """, unsafe_allow_html=True)

# --- 4. 主程式 ---
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
    except: st.error("⚠️ 數據庫同步中，請稍候..."); return

    sd = {i['setting_name']: i['value'] for i in ws_s.get_all_records()}
    cp, ct = int(sd.get('global_precision', 55)), int(sd.get('api_ttl_min', 5))

    if not st.session_state.user:
        st.title("🚀 StockAI 登入")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("確認登入"):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = vault["u"] = u; st.rerun()
            else: st.error("❌ 驗證失敗")
    else:
        # 修正後的控制面板 (文字全亮化)
        with st.expander("⚙️ 終端功能面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_s = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的選股清單", user_s if user_s else ["2330.TW"])
                if st.button("🚪 登出系統"): st.session_state.user = vault["u"] = None; st.rerun()
            with m2:
                p_days = st.number_input("AI 預測天數 (最大 30 日)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    npw = st.slider("全域靈敏度控制", 0, 100, cp)
                    nt = st.select_slider("API 更新時間", options=list(range(1, 11)), value=ct)
                    if st.button("💾 儲存並同步設定"):
                        ws_s.update_cell(2, 2, str(npw)); ws_s.update_cell(3, 2, str(nt)); st.rerun()
        show_dashboard(target, p_days, cp)

if __name__ == "__main__": main()
