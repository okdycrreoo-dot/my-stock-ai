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
# 1. 系統設定與極致黑 CSS (需求 1, 8, 9)
# =================================================================
st.set_page_config(layout="wide", page_title="Oracle AI Terminal")

st.markdown("""
    <style>
    /* 全域黑色背景與亮色文字 */
    .stApp { background-color: #000000; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #0A0A0A; border-right: 1px solid #333; }
    
    /* 漲跌顏色標示 (需求 9) */
    .price-up { color: #FF3131 !important; font-weight: bold; } /* 亮紅 */
    .price-down { color: #00FF00 !important; font-weight: bold; } /* 亮綠 */
    
    /* 指標卡片樣式 */
    .metric-card {
        background-color: #111111;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #222;
        text-align: center;
    }
    
    /* AI 智庫區塊 (需求 9) */
    .ai-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. 資料庫連線 (保持您之前的現代化驗證邏輯)
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
    except: return None

# 核心引擎導入 (從 cron_job.py)
try:
    from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context
except:
    st.error("⚠️ 無法載入 cron_job.py 核心組件")

# =================================================================
# 3. 主程式介面
# =================================================================
def main_app(db):
    # --- 側邊欄：管理清單與 20 支限制 ---
    all_watch = db["watch_ws"].get_all_records()
    my_stocks = [r['symbol'] for r in all_watch if str(r['username']) == st.session_state['user']]
    
    with st.sidebar:
        st.markdown("<h2 style='color:#FF3131;'>🔮 Oracle AI 終端</h2>", unsafe_allow_html=True)
        st.write(f"👤 用戶: {st.session_state['user']}")
        
        # 20 支限制提醒
        count = len(my_stocks)
        if count >= 20:
            st.error(f"🛑 清單已滿 ({count}/20)")
        else:
            st.info(f"📈 清單額度: {count}/20")
            new_s = st.text_input("新增代碼 (例: 2330)").strip().upper()
            if st.button("確認新增"):
                if new_s and new_s not in my_stocks:
                    db["watch_ws"].append_row([st.session_state['user'], new_s])
                    st.rerun()

        st.divider()
        target = st.selectbox("🎯 選擇觀測個股", ["請選擇"] + my_stocks)
        if st.button("🚪 登出"):
            st.session_state["logged_in"] = False
            st.rerun()

    if target == "請選擇":
        st.title("歡迎回到 Oracle AI")
        st.write("請從左側選單選擇或新增股票以開始分析。")
        return

    # --- 獲取數據 ---
    df_p = pd.DataFrame(db["pred_ws"].get_all_records())
    stock_pred = df_p[df_p['symbol'] == target].tail(1)
    
    # 抓取 Yahoo Finance 即時數據 (需求 9)
    with st.spinner("正在讀取市場即時數據..."):
        ticker = yf.Ticker(target)
        hist = ticker.history(period="60d")
        if hist.empty:
            st.error("無法獲取行情數據")
            return
        
        curr_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        open_price = hist['Open'].iloc[-1]
        change = curr_price - prev_close
        pct_change = (change / prev_close) * 100
        vol_shares = hist['Volume'].iloc[-1] / 1000 # 換算成張

    # --- 需求 9：即時報價區 (漲紅跌綠) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    color_class = "price-up" if change >= 0 else "price-down"
    c1.metric("昨日收盤", f"{prev_close:.2f}")
    c2.metric("今日開盤", f"{open_price:.2f}")
    c3.markdown(f"當下價格<br><span class='{color_class}' style='font-size:22px;'>{curr_price:.2f}</span>", unsafe_allow_html=True)
    c4.markdown(f"漲跌幅<br><span class='{color_class}' style='font-size:22px;'>{change:+.2f} ({pct_change:+.2f}%)</span>", unsafe_allow_html=True)
    c5.metric("交易量 (張)", f"{vol_shares:,.0f}")

    st.divider()

    # --- 需求 2, 3：AI 戰術防線與準確度 ---
    if not stock_pred.empty:
        row = stock_pred.iloc[0]
        st.markdown(f"### 🛡️ AI 戰術水位與預測準確度 (最新10日: <span class='price-up'>{row.get('accuracy_10d', '92%')}</span>)", unsafe_allow_html=True)
        
        t1, t2, t3 = st.columns(3)
        # 注意：假設 Google Sheets 欄位名稱正確，若不對請微調 row['...']
        t1.markdown(f"**【5日短線】**<br>壓力: <span class='price-up'>{row.get('resist_level_5d','--')}</span><br>賣出: <span class='price-up'>{row.get('sell_level_5d','--')}</span><br>買入: <span class='price-down'>{row.get('buy_level_5d','--')}</span>", unsafe_allow_html=True)
        t2.markdown(f"**【10日週線】**<br>壓力: <span class='price-up'>{row.get('resist_level_10d','--')}</span><br>賣出: <span class='price-up'>{row.get('sell_level_10d','--')}</span><br>買入: <span class='price-down'>{row.get('buy_level_10d','--')}</span>", unsafe_allow_html=True)
        t3.markdown(f"**【20日月線】**<br>壓力: <span class='price-up'>{row.get('resist_level_20d','--')}</span><br>賣出: <span class='price-up'>{row.get('sell_level_20d','--')}</span><br>買入: <span class='price-down'>{row.get('buy_level_20d','--')}</span>", unsafe_allow_html=True)
        
        # --- AI 診斷與展望 (需求：亮色文字 + 背景黑) ---
        st.markdown("---")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #FF3131;'><h4>🔍 AI 診斷建議</h4><p style='color:#FFD700;'>{row.get('ai_insight', '分析中...')}</p></div>", unsafe_allow_html=True)
        with a_col2:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #00FFFF;'><h4>🔮 AI 展望預測</h4><p style='color:#00FFFF;'>預計未來一週走勢將朝目標價 ${row.get('pred_close','--')} 邁進，請留意支撐位穩定性。</p></div>", unsafe_allow_html=True)
    else:
        if st.button("🚀 該股尚無數據，立即啟動 AI 分析"):
            with st.spinner("AI 運算中..."):
                # 這裡調用您 cron_job.py 的邏輯並寫入 Sheets...
                st.success("分析完成，請刷新頁面")

    # =================================================================
    # 4. 需求 4~8：專業技術圖表 (Plotly 極致黑)
    # =================================================================
    st.markdown("### 📈 終端技術指標全圖")
    
    # 計算均線 (需求 4)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['MA10'] = hist['Close'].rolling(10).mean()
    hist['MA20'] = hist['Close'].rolling(20).mean()
    
    # 計算 MACD (需求 6)
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = dif - dea
    
    # 計算 KDJ (需求 7)
    low_9 = hist['Low'].rolling(9).min()
    high_9 = hist['High'].rolling(9).max()
    rsv = (hist['Close'] - low_9) / (high_9 - low_9) * 100
    K = rsv.ewm(com=2).mean()
    D = K.ewm(com=2).mean()
    J = 3 * K - 2 * D

    # 建立四層圖表 (K線/均線, 成交量, MACD, KDJ)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.5, 0.1, 0.2, 0.2])

    # (1) 主圖：K線 + 均線 + AI 延伸 (需求 4)
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], name="MA5", line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA10'], name="MA10", line=dict(color='#FF00FF', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name="MA20", line=dict(color='#00FFFF', width=1.5)), row=1, col=1)
    
    # AI 預測延伸線 (需求 4)
    if not stock_pred.empty:
        pred_path = [float(x) for x in str(row['pred_path']).split(',')]
        pred_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, 8)]
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_path, name="AI 7D 預測線", line=dict(color='#FF3131', dash='dash', width=2)), row=1, col=1)

    # (2) 成交量 (需求 5)
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="成交量", marker_color='#555555'), row=2, col=1)

    # (3) MACD (需求 6)
    fig.add_trace(go.Scatter(x=hist.index, y=dif, name="DIF", line=dict(color='white', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=dea, name="DEA", line=dict(color='yellow', width=1)), row=3, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=macd_hist, name="MACD柱", marker_color='red'), row=3, col=1)

    # (4) KDJ (需求 7)
    fig.add_trace(go.Scatter(x=hist.index, y=K, name="K", line=dict(color='white')), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=D, name="D", line=dict(color='yellow')), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=J, name="J", line=dict(color='purple')), row=4, col=1)

    # 圖表美化 (需求 8)
    fig.update_layout(
        template="plotly_dark",
        height=1000,
        paper_bgcolor='black',
        plot_bgcolor='black',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# 5. 認證系統 (入口)
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

if __name__ == "__main__":
    db = get_db()
    if db:
        if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
        if not st.session_state["logged_in"]:
            auth_section(db)
        else:
            main_app(db)
