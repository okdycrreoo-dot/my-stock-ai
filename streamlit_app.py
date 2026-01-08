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

# --- 1. 配置與 UI 深度修復 ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")
ICON_URL = "https://raw.githubusercontent.com/okdycrreoo-dot/my-stock-ai/main/icon.png"

st.markdown(f"""
    <link rel="icon" type="image/png" href="{ICON_URL}">
    <style>
    /* 全域文字與背景設定 */
    .stApp {{ background-color: #0E1117; color: #FFFFFF !important; }}
    
    /* 強制修正控制面板文字顏色 (Label, Slider, Input) */
    .stMarkdown, p, label, .stSlider label, .stNumberInput label {{ 
        color: #FFFFFF !important; 
        font-weight: 700 !important; 
        font-size: 1.05rem !important; 
    }}
    
    /* 下拉選單與輸入框文字 */
    div[data-baseweb="select"] div, input {{ 
        color: #FFFFFF !important; 
        background-color: #1C2128 !important; 
    }}
    
    /* 按鈕文字顏色修正 */
    .stButton>button {{ 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border-radius: 12px; 
        font-weight: 900 !important; 
        width: 100%;
        border: none;
    }}
    
    /* 資訊方塊與新聞卡片 */
    .diag-box {{ background-color: #161B22; border-left: 5px solid #00F5FF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #30363D; }}
    .news-card {{ background-color: #1C2128; border-radius: 8px; padding: 15px; margin-bottom: 12px; border-left: 5px solid #FF4500; border: 1px solid #30363D; }}
    .price-tag {{ font-size: 1.25rem; font-weight: 800; }}
    
    /* 隱藏側邊欄按鈕 */
    button[data-testid="sidebar-button"] {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據與新聞引擎 ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="2y", interval="1d", auto_adjust=True)
        if data.empty: return None, []
        
        # 指標計算
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
        data['V_MA5'] = data['Volume'].rolling(5).mean()
        
        # 修正新聞抓取：使用更穩定的 Search 功能
        search = yf.Search(symbol, max_results=5)
        news_list = search.news
        
        return data.dropna(), news_list
    except: return None, []

# --- 3. AI 總結診斷 ---
def get_ai_summary(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 50
    reasons = []
    if last['Hist'] > prev['Hist']: score += 15; reasons.append("MACD 動能向上")
    else: score -= 10; reasons.append("MACD 動能走平")
    if last['J'] < 20: score += 10; reasons.append("KDJ 超賣預警")
    elif last['J'] > 80: score -= 10; reasons.append("KDJ 超買預警")
    status = "🚀 看多" if score > 65 else ("📉 看空" if score < 40 else "⚖️ 震盪")
    return status, reasons

# --- 4. 繪圖與展示 ---
def show_dashboard(symbol, p_days, precision):
    df, news = fetch_comprehensive_data(symbol)
    if df is None: st.error("⚠️ 無法獲取股票數據，請確認代碼。"); return
    
    # AI 買賣價看板
    bias = (int(precision) - 55) / 100
    st.markdown("### 🤖 AI 智能多指標分析")
    c1, c2, c3 = st.columns(3)
    p_pts = {"5日": 0.03, "20日": 0.06, "60日": 0.10}
    for i, (k, v) in enumerate(p_pts.items()):
        ma_val = df[f'MA{k[:-1]}'].iloc[-1]
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='diag-box'><b>{k}區間</b><br>🟢 買: <span class='price-tag' style='color:#00FF41'>{(ma_val*(1-v+bias)):.2f}</span><br>🔴 賣: <span class='price-tag' style='color:#FF3131'>{(ma_val*(1+v+bias)):.2f}</span></div>", unsafe_allow_html=True)

    # 市場診斷
    status, reasons = get_ai_summary(df)
    st.info(f"**市場診斷：{status}** | 解析：{', '.join(reasons)}")

    # 四層圖表
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
    p_df = df.tail(60)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量', marker_color=['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]), 2, 1)
    # MACD 青色加粗
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2.5)), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], name='Signal', line=dict(color='#FFFF00', width=1.2)), 3, 1)
    # KDJ 綠色加粗
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K線', line=dict(color='#00FF41', width=2.5)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J線', line=dict(color='#FF00FF', width=1.2)), 4, 1)
    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 補回：市場新聞區
    st.markdown("### 📰 相關市場新聞")
    if not news: st.warning("目前暫無相關新聞。")
    for n in news:
        st.markdown(f"""<div class='news-card'><b style='color:#00F5FF; font-size:1.1rem;'>{n.get('title','')}</b><br><small style='color:#AAAAAA;'>{n.get('publisher','')} | {datetime.fromtimestamp(n.get('providerPublishTime',0)).strftime('%Y-%m-%d %H:%M')}</small><br><a href='{n.get('link','#')}' target='_blank' style='color:#FF4500; text-decoration:none; font-weight:bold;'>點此閱讀全文 →</a></div>""", unsafe_allow_html=True)

# --- 5. 主程式 ---
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
    except: st.error("⚠️ 數據庫連線異常"); return

    sd = {i['setting_name']: i['value'] for i in ws_s.get_all_records()}
    cp, ct = int(sd.get('global_precision', 55)), int(sd.get('api_ttl_min', 5))

    if not st.session_state.user:
        st.title("🚀 StockAI 登入")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("確認登入"):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = vault["u"] = u; st.rerun()
            else: st.error("❌ 帳號密碼錯誤")
    else:
        with st.expander("⚙️ 終端功能面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_s = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的選股", user_s if user_s else ["2330.TW"])
                if st.button("🚪 安全登出系統"): st.session_state.user = vault["u"] = None; st.rerun()
            with m2:
                p_days = st.number_input("AI 預測天數 (最大 30 日)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    npw = st.slider("全域靈敏度", 0, 100, cp)
                    nt = st.select_slider("API 連線時間 (分)", options=list(range(1, 11)), value=ct)
                    if st.button("💾 同步資料庫設定"):
                        ws_s.update_cell(2, 2, str(npw)); ws_s.update_cell(3, 2, str(nt)); st.rerun()
        show_dashboard(target, p_days, cp)

if __name__ == "__main__": main()
