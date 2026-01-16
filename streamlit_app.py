import streamlit as st
import pandas as pd
import numpy as np
import json
import gspread
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz
import time

# =================================================================
# 1. 系統設定與極致黑 CSS
# =================================================================
st.set_page_config(layout="wide", page_title="Oracle AI Terminal")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #0A0A0A; border-right: 1px solid #333; }
    .price-up { color: #FF3131 !important; font-weight: bold; } 
    .price-down { color: #00FF00 !important; font-weight: bold; } 
    .metric-card {
        background-color: #111111;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #222;
        text-align: center;
    }
    .ai-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #333;
        background-color: #0A0A0A;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. 資料庫連線
# =================================================================
@st.cache_resource
def get_db():
    creds_info = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_info: return None
    try:
        info = json.loads(creds_info)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(info, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open("users")
        return {
            "user_ws": sh.worksheet("users"),
            "watch_ws": sh.worksheet("watchlist"),
            "pred_ws": sh.worksheet("predictions")
        }
    except Exception as e:
        st.error(f"資料庫連線失敗: {e}")
        return None

# =================================================================
# 3. 主程式介面 (對齊 37 欄位與 20 支限制)
# =================================================================
def main_app(db):
    # --- 側邊欄：管理與 20 支限制 ---
    all_watch = db["watch_ws"].get_all_records()
    my_stocks = [r['symbol'] for r in all_watch if str(r['username']) == st.session_state['user']]
    
    with st.sidebar:
        st.markdown("<h2 style='color:#FF3131;'>🔮 Oracle AI 終端</h2>", unsafe_allow_html=True)
        st.write(f"👤 用戶: {st.session_state['user']}")
        
        # 執行 20 支限制
        count = len(my_stocks)
        if count >= 20:
            st.error(f"🛑 監控清單已滿 ({count}/20)")
            new_s = st.text_input("新增代碼 (已達上限)", disabled=True)
        else:
            st.info(f"📈 清單額度: {count}/20")
            new_s = st.text_input("新增代碼 (例: 2330)").strip().upper()
            if st.button("確認新增"):
                if new_s and new_s not in my_stocks:
                    db["watch_ws"].append_row([st.session_state['user'], new_s])
                    st.success(f"{new_s} 已加入")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        target = st.selectbox("🎯 選擇觀測個股", ["請選擇"] + my_stocks)
        if st.button("🚪 登出"):
            st.session_state["logged_in"] = False
            st.rerun()

    if target == "請選擇":
        st.title("歡迎回到 Oracle AI")
        st.write("請從左側選單選擇個股。")
        return

    # --- 獲取數據與 37 欄位對齊 ---
    df_p = pd.DataFrame(db["pred_ws"].get_all_records())
    stock_pred = df_p[df_p['symbol'] == target].tail(1)
    
    with st.spinner("讀取市場數據中..."):
        ticker = yf.Ticker(target)
        hist = ticker.history(period="60d")
        if hist.empty:
            st.error("無法獲取行情")
            return
        
        curr_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = curr_price - prev_close
        pct_change = (change / prev_close) * 100

    # (A) 即時報價區
    color_class = "price-up" if change >= 0 else "price-down"
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    c1.metric("昨日收盤", f"{prev_close:.2f}")
    c2.metric("即時價格", f"{curr_price:.2f}")
    c3.markdown(f"漲跌幅 <br><span class='{color_class}' style='font-size:24px;'>{change:+.2f} ({pct_change:+.2f}%)</span>", unsafe_allow_html=True)
    
    # AK 欄位：市場情緒
    if not stock_pred.empty:
        sentiment = stock_pred.iloc[0].get('market_sentiment', '穩定')
        c4.metric("AI 市場情緒 (AK)", sentiment)

    st.divider()

    # (B) AI 神之大腦核心 (37 欄位展現)
    if not stock_pred.empty:
        row = stock_pred.iloc[0]
        
        # 診斷與展望 (AB, AC 欄位)
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #FF3131;'><h4>🔍 Oracle 診斷 (AB)</h4><p style='color:#FFD700;'>{row.get('ai_insight', '分析中...')}</p></div>", unsafe_allow_html=True)
        with a_col2:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #00FFFF;'><h4>🔮 AI 展望 (AC)</h4><p style='color:#00FFFF;'>{row.get('forecast_outlook', '計算中...')}</p></div>", unsafe_allow_html=True)

        # 戰略水位矩陣 (G-X 欄位)
        st.markdown("### 🛡️ 戰略水位 (G-X 18 欄位精確對齊)")
        t1, t2, t3 = st.columns(3)
        t1.markdown(f"**【支撐買點 (Buy)】**<br>5D: {row.get('buy_level_5d','--')}<br>10D: {row.get('buy_level_10d','--')}<br>20D: {row.get('buy_level_20d','--')}", unsafe_allow_html=True)
        t2.markdown(f"**【壓力賣點 (Sell)】**<br>5D: {row.get('sell_level_5d','--')}<br>10D: {row.get('sell_level_10d','--')}<br>20D: {row.get('sell_level_20d','--')}", unsafe_allow_html=True)
        t3.markdown(f"**【強力反轉 (Resist)】**<br>5D: {row.get('resist_level_5d','--')}<br>10D: {row.get('resist_level_10d','--')}<br>20D: {row.get('resist_level_20d','--')}", unsafe_allow_html=True)

        # 專家指標 (AH-AJ)
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATR 波動 (AH)", row.get('atr_val', '--'))
        m2.metric("量比 (AI)", row.get('volume_ratio', '--'))
        m3.metric("盈虧比 (AJ)", row.get('risk_reward', '--'))
        m4.metric("5D 乖離 (AD)", f"{row.get('bias_5d', '--')}%")

    # (C) 專業技術圖表 (K線 + AI 7D 預測路徑 AA)
    st.markdown("### 📈 終端技術指標全圖 (對齊 AA 預測路徑)")
    
    # 計算指標
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['MA20'] = hist['Close'].rolling(20).mean()
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = dif - dea

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])

    # K線
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], name="MA5", line=dict(color='#FFD700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name="MA20", line=dict(color='#00FFFF')), row=1, col=1)

    # AA 欄位：AI 7日預測延伸
    if not stock_pred.empty and row.get('pred_path'):
        try:
            pp = [float(x) for x in str(row['pred_path']).split(',')]
            p_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, 8)]
            fig.add_trace(go.Scatter(x=p_dates, y=pp, name="AI 7D 預測", line=dict(color='#FF3131', dash='dash')), row=1, col=1)
        except: pass

    # 成交量
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="成交量", marker_color='#333333'), row=2, col=1)
    # MACD
    fig.add_trace(go.Bar(x=hist.index, y=macd_hist, name="MACD"), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=800, paper_bgcolor='black', plot_bgcolor='black', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# 4. 認證系統 (入口)
# =================================================================
def auth_section(db):
    st.markdown("<h1 style='text-align: center; color: #FF3131;'>🔮 ORACLE AI SYSTEM</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["登入系統", "註冊帳號"])
    
    with tab1:
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        if st.button("啟動終端"):
            users = db["user_ws"].get_all_records()
            found = next((row for row in users if str(row['username'])==u and str(row['password'])==p), None)
            if found:
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.rerun()
            else: st.error("認證失敗")
            
    with tab2:
        new_u = st.text_input("新帳號")
        new_p = st.text_input("新密碼", type="password")
        if st.button("建立權限"):
            if new_u and new_p:
                db["user_ws"].append_row([new_u, new_p])
                st.success("註冊成功，請切換至登入分頁")

if __name__ == "__main__":
    db = get_db()
    if db:
        if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
        if not st.session_state["logged_in"]:
            auth_section(db)
        else:
            main_app(db)
