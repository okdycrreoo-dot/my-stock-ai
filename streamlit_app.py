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

# --- 1. 配置與 UI 視覺深度強化 (解決面板文字隱沒問題) ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    /* 1. 全域背景與文字亮度強化 */
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
        text-shadow: 1px 1px 2px #000000;
    }
    
    /* 2. 修正展開面板樣式 */
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #00F5FF !important;
        border-radius: 12px !important;
        font-weight: 900 !important;
    }

    /* 3. 輸入框與下拉選單 */
    div[data-baseweb="select"] > div { 
        background-color: #1C2128 !important; 
        color: #FFFFFF !important; 
        border: 2px solid #00F5FF !important; 
    }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }

    /* 4. 青色按鈕 */
    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border: 2px solid #FFFFFF !important;
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important;
    }
    
    /* 5. 診斷盒與價格標籤 */
    .diag-box { 
        background-color: #161B22; 
        border-left: 6px solid #00F5FF; 
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 15px; 
        border: 1px solid #30363D; 
    }
    .price-tag { font-size: 1.4rem; font-weight: 900; }
    
    /* 隱藏原生側欄 */
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (台股優化 & 自動後綴) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_tw_stock_data(symbol):
    # 台股代碼自動處理 (.TW / .TWO)
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")):
        final_symbol = f"{s}.TW"
    else:
        final_symbol = s

    for _ in range(3):
        try:
            ticker = yf.Ticker(final_symbol)
            # 要求 6: 獲取股票名稱
            stock_name = ticker.info.get('longName') or ticker.info.get('shortName') or final_symbol
            
            # 若獲取不到數據且是 .TW，嘗試切換為 .TWO (上櫃)
            data = ticker.history(period="2y", interval="1d", auto_adjust=True)
            if (data is None or data.empty) and ".TW" in final_symbol:
                final_symbol = final_symbol.replace(".TW", ".TWO")
                ticker = yf.Ticker(final_symbol)
                stock_name = ticker.info.get('longName') or final_symbol
                data = ticker.history(period="2y", interval="1d", auto_adjust=True)

            if data is not None and not data.empty:
                # 指標計算
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA60'] = data['Close'].rolling(60).mean()
                # MACD
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                # 要求 3: KDJ 計算
                low_9 = data['Low'].rolling(9).min()
                high_9 = data['High'].rolling(9).max()
                rsv = (data['Close'] - low_9) / (high_9 - low_9) * 100
                data['K'] = rsv.ewm(com=2, adjust=False).mean()
                data['D'] = data['K'].ewm(com=2, adjust=False).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                
                # 市場焦點摘要
                search = yf.Search(final_symbol, max_results=3)
                news = " | ".join([n['title'] for n in search.news]) if search.news else "目前市場無重大訊息。"
                
                return data.dropna(), stock_name, news, final_symbol
            time.sleep(1.2)
        except:
            time.sleep(1.2); continue
    return None, symbol, "無數據", symbol

# --- 3. AI 綜合分析與買賣建議價 (要求 4) ---
def perform_tw_ai_analysis(df, precision, stock_name, news):
    last, prev = df.iloc[-1], df.iloc[-2]
    # 考量管理員靈敏度
    bias = (int(precision) - 55) / 100
    
    # 權重分析：MACD、成交量、KDJ、市場評分
    macd_slope = 1.02 if last['Hist'] > prev['Hist'] else 0.98
    vol_slope = 1.01 if last['Volume'] > df['Volume'].tail(5).mean() else 0.99
    k_slope = 1.03 if last['K'] < 25 else (0.97 if last['K'] > 75 else 1.0)
    
    total_mod = macd_slope * vol_slope * k_slope + bias
    
    # 計算買賣建議價 (基於月線 MA20 之偏離回歸)
    buy_p = last['MA20'] * 0.96 * total_mod
    sell_p = last['MA20'] * 1.05 * total_mod
    
    score = 50
    if last['Close'] > last['MA20']: score += 15
    if last['Hist'] > 0: score += 10
    
    return {
        "name": stock_name,
        "buy": buy_p,
        "sell": sell_p,
        "score": score,
        "news": news
    }

# --- 4. 儀表板繪製層 (要求 1, 3, 6) ---
def show_ultimate_dashboard(symbol, p_days, precision):
    df, full_name, news_txt, final_id = fetch_tw_stock_data(symbol)
    if df is None: st.error(f"❌ 無法讀取股票代碼 '{symbol}'"); return

    ai = perform_tw_ai_analysis(df, precision, full_name, news_txt)
    
    # 要求 6: 顯示股票名稱與代碼
    st.title(f"🇹🇼 {ai['name']} ({final_id})")
    
    # 要求 4: 買賣建議價
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='diag-box'>🟢 AI 建議買入價<br><span class='price-tag' style='color:#00FF41'>{ai['buy']:.2f}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='diag-box'>🔴 AI 建議賣出價<br><span class='price-tag' style='color:#FF3131'>{ai['sell']:.2f}</span></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='diag-box'>⚖️ AI 綜合評分<br><span class='price-tag' style='color:#00F5FF'>{ai['score']}</span></div>", unsafe_allow_html=True)

    # 四層結構圖表 (要求 1: 移除單位默認日線, 要求 3: 加入 KDJ 並明顯化)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.45, 0.15, 0.15, 0.25], vertical_spacing=0.03)
    p_df = df.tail(60) # 默認固定日線範圍
    
    # Layer 1: K線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    # Layer 2: 成交量
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    # Layer 3: MACD (青色)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2)), 3, 1)
    # Layer 4: KDJ (要求 3: 綠線加粗)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(綠加粗)', line=dict(color='#00FF41', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J線', line=dict(color='#FF00FF', width=1.2)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)

    # 市場新聞分析 (摘要版)
    st.markdown("### 📰 AI 市場重點解析")
    st.info(f"📌 **本日焦點**：{ai['news']}\n\n💡 **AI 建議**：結合台股籌碼與 KDJ 指標，{ai['name']} 當前支撐建議觀察 {ai['buy']:.2f}，壓力位階約為 {ai['sell']:.2f}。操作上宜分批布局。")

# --- 5. 主程式與持久化登入 (要求 5) ---
def main():
    # 要求 5: 30分鐘持久化登入鎖
    @st.cache_resource(ttl=1800)
    def auth_vault(): return {"user": None}
    
    session_vault = auth_vault()
    if 'user' not in st.session_state: st.session_state.user = session_vault["user"]

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: st.error("⚠️ 雲端資料庫連線中..."); return

    sd = {i['setting_name']: i['value'] for i in ws_s.get_all_records()}
    cp = int(sd.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🚀 StockAI 台股終端登入")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("啟動系統", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = session_vault["user"] = u; st.rerun()
            else: st.error("❌ 帳密驗證失敗")
    else:
        with st.expander("⚙️ 終端管理面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的台股清單", user_stocks if user_stocks else ["2330.TW"])
                if st.button("🚪 安全登出系統"):
                    st.session_state.user = session_vault["user"] = None; st.rerun()
            with m2:
                # 要求 2: 文字標註最大值 30
                p_days = st.number_input("AI 預測天數 (最大值30)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("管理員靈敏度", 0, 100, cp)
                    if st.button("💾 同步雲端設定"):
                        ws_s.update_cell(2, 2, str(new_p)); st.rerun()
        
        show_ultimate_dashboard(target, p_days, cp)

if __name__ == "__main__": main()
