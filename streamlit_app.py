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

# --- 1. 配置與 UI 視覺深度修復 ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    /* 全域背景與文字亮度強化 */
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    
    /* 標籤文字全白加粗 */
    label, p, span, .stMarkdown, .stCaption { 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
        text-shadow: 1px 1px 2px #000000;
    }
    
    /* 展開面板樣式 */
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #00F5FF !important;
        border-radius: 12px !important;
    }

    /* 輸入框與下拉選單 */
    div[data-baseweb="select"] > div { 
        background-color: #1C2128 !important; 
        color: #FFFFFF !important; 
        border: 2px solid #00F5FF !important; 
    }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }

    /* 青色按鈕 */
    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important;
    }
    
    /* AI 建議與診斷區塊 */
    .diag-box { 
        background-color: #161B22; 
        border-left: 6px solid #00F5FF; 
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 15px; 
        border: 1px solid #30363D; 
    }
    .price-tag { font-size: 1.4rem; font-weight: 900; }
    .summary-card { 
        background-color: #1C2128; 
        border-radius: 10px; 
        padding: 20px; 
        border: 1px solid #00F5FF; 
        border-left: 8px solid #FF4500; 
    }
    
    /* 隱藏側欄按鈕 */
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 台股數據引擎 (自動補全後綴) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_tw_stock_data(symbol):
    # 台股代碼自動校正邏輯
    clean_symbol = str(symbol).strip().upper()
    if not clean_symbol.endswith(".TW") and not clean_symbol.endswith(".TWO"):
        # 預設嘗試上市 (.TW)，若無則嘗試上櫃 (.TWO)
        final_symbol = f"{clean_symbol}.TW"
    else:
        final_symbol = clean_symbol

    for attempt in range(3):
        try:
            ticker = yf.Ticker(final_symbol)
            info = ticker.info
            
            # 若獲取不到名稱，嘗試切換上櫃後綴
            if not info.get('longName') and ".TW" in final_symbol:
                final_symbol = final_symbol.replace(".TW", ".TWO")
                ticker = yf.Ticker(final_symbol)
                info = ticker.info

            data = ticker.history(period="2y", interval="1d", auto_adjust=True)
            if data is not None and not data.empty:
                # 1. 技術指標計算 (MA, MACD, KDJ)
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA60'] = data['Close'].rolling(60).mean()
                
                # MACD
                ema12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema26 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = ema12 - ema26
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                
                # KDJ (要求3)
                low_9 = data['Low'].rolling(9).min()
                high_9 = data['High'].rolling(9).max()
                rsv = (data['Close'] - low_9) / (high_9 - low_9) * 100
                data['K'] = rsv.ewm(com=2, adjust=False).mean()
                data['D'] = data['K'].ewm(com=2, adjust=False).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                
                data['V_MA5'] = data['Volume'].rolling(5).mean()
                
                # 台股新聞摘要
                search = yf.Search(final_symbol, max_results=3)
                news_summary = " | ".join([n.get('title') for n in search.news]) if search.news else "查無相關台股即時新聞。"
                
                return data.dropna(), info.get('longName', symbol), news_summary, final_symbol
            time.sleep(1.2)
        except:
            time.sleep(1.2); continue
    return None, symbol, "獲取數據失敗", final_symbol

# --- 3. AI 綜合分析與建議價 (要求4) ---
def perform_ai_analysis(df, precision, stock_name, news_txt):
    last, prev = df.iloc[-1], df.iloc[-2]
    bias = (int(precision) - 55) / 100
    
    # 權重計分邏輯
    macd_val = 1.02 if last['Hist'] > prev['Hist'] else 0.98
    vol_val = 1.01 if last['Volume'] > last['V_MA5'] else 0.99
    kdj_val = 1.03 if last['K'] < 20 else (0.97 if last['K'] > 80 else 1.0)
    
    total_mod = macd_val * vol_val * kdj_val + bias
    
    # 計算買賣建議價
    buy_p = last['MA20'] * 0.97 * total_mod
    sell_p = last['MA20'] * 1.06 * total_mod
    
    # 診斷得分
    score = 50
    if last['Close'] > last['MA20']: score += 15
    if last['Hist'] > 0: score += 10
    if last['K'] < 30: score += 10 # 低檔起漲
    
    return {"name": stock_name, "buy": buy_p, "sell": sell_p, "score": score, "news": news_txt}

# --- 4. 儀表板繪製層 (台股優化版) ---
def show_tw_dashboard(symbol, p_days, precision):
    df, full_name, news_summary, final_id = fetch_tw_stock_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取台股代碼 '{symbol}'。提示：請輸入 2330 或 2330.TW"); return

    ai = perform_ai_analysis(df, precision, full_name, news_summary)
    
    # 顯示股票名稱與代號 (要求6)
    st.title(f"🇹🇼 {ai['name']} ({final_id})")
    
    # 顯示建議價 (要求4)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='diag-box'>🟢 AI 建議買入價<br><span class='price-tag' style='color:#00FF41'>{ai['buy']:.2f}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='diag-box'>🔴 AI 建議賣出價<br><span class='price-tag' style='color:#FF3131'>{ai['sell']:.2f}</span></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='diag-box'>⚖️ AI 綜合評分<br><span class='price-tag' style='color:#00F5FF'>{ai['score']}</span></div>", unsafe_allow_html=True)

    # 四層結構圖表 (要求1: 日線, 要求3: KDJ)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.45, 0.15, 0.15, 0.25], vertical_spacing=0.03)
    p_df = df.tail(60) 
    
    # Layer 1: K線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    # Layer 2: 成交量
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    # Layer 3: MACD (青色)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2)), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], name='訊號', line=dict(color='#FFFF00', width=1)), 3, 1)
    # Layer 4: KDJ (要求3: 綠線加粗)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(綠加粗)', line=dict(color='#00FF41', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J', line=dict(color='#FF00FF', width=1.2)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)

    # AI 總結摘要
    st.markdown("### 📰 台股市場重點解析")
    st.markdown(f"""
        <div class='summary-card'>
            <p style='font-size:1.1rem;'>{ai['news']}</p>
            <hr style='border: 0.5px solid #30363D;'>
            <p style='color:#00F5FF;'><b>💡 AI 操作指引：</b> 根據當前台股籌碼與指標，建議於 {ai['buy']:.2f} 附近觀察支撐，若站穩可看至目標價 {ai['sell']:.2f}。</p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. 主程式與持久化登入 (要求5) ---
def main():
    @st.cache_resource(ttl=1800) # 要求5: 30分鐘長效
    def persistent_auth(): return {"user": None}
    
    vault = persistent_auth()
    if 'user' not in st.session_state: st.session_state.user = vault["user"]

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: st.error("⚠️ 數據庫連線異常"); return

    sd = {i['setting_name']: i['value'] for i in ws_s.get_all_records()}
    cp = int(sd.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🇹🇼 StockAI 台股終端登入")
        u_in = st.text_input("帳號")
        p_in = st.text_input("密碼", type="password")
        if st.button("啟動系統", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u_in) & (udf['password'].astype(str)==p_in)].empty:
                st.session_state.user = vault["user"] = u_in; st.rerun()
            else: st.error("❌ 帳號密碼錯誤")
    else:
        with st.expander("⚙️ 終端管理面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的台股清單", user_stocks if user_stocks else ["2330.TW"])
                if st.button("🚪 安全登出系統"):
                    st.session_state.user = vault["user"] = None; st.rerun()
            with m2:
                # 要求2: 預測天數最大值30
                p_days = st.number_input("AI 預測天數 (最大值30)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("全域靈敏度", 0, 100, cp)
                    if st.button("💾 同步設定"): ws_s.update_cell(2, 2, str(new_p)); st.rerun()
        
        show_tw_dashboard(target, p_days, cp)

if __name__ == "__main__": main()
