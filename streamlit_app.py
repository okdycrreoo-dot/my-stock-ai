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
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

# 視覺增強樣式表
st.markdown("""
    <style>
    /* 全域背景與文字亮度強化 */
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    
    /* 控制面板文字全白加粗 */
    label, p, span, .stMarkdown, .stCaption { 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
        text-shadow: 1px 1px 2px #000000;
    }
    
    /* 修正展開面板標題 */
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #00F5FF !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
    }

    /* 下拉選單與輸入框 */
    div[data-baseweb="select"] > div { 
        background-color: #1C2128 !important; 
        color: #FFFFFF !important; 
        border: 2px solid #00F5FF !important; 
    }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }

    /* 青色按鈕樣式 */
    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important;
        border: 1px solid #FFFFFF !important;
    }
    
    /* 診斷卡片與數值文字 */
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
    
    /* 隱藏原生側欄按鈕 */
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據與新聞引擎 (強化穩定性) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            # 獲取完整名稱 (要求6)
            info = ticker.info
            full_name = info.get('longName') or info.get('shortName') or symbol
            
            data = ticker.history(period="2y", interval="1d", auto_adjust=True)
            if data is not None and not data.empty:
                # 1. 移動平均線
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA60'] = data['Close'].rolling(60).mean()
                
                # 2. MACD 計算
                ema12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema26 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = ema12 - ema26
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                
                # 3. KDJ 計算 (要求3)
                low_9 = data['Low'].rolling(9).min()
                high_9 = data['High'].rolling(9).max()
                rsv = (data['Close'] - low_9) / (high_9 - low_9) * 100
                data['K'] = rsv.ewm(com=2, adjust=False).mean()
                data['D'] = data['K'].ewm(com=2, adjust=False).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                
                # 4. 成交量均線
                data['V_MA5'] = data['Volume'].rolling(5).mean()
                
                # 5. 獲取新聞重點
                search = yf.Search(symbol, max_results=3)
                news_summary = " | ".join([n.get('title') for n in search.news]) if search.news else "目前市場無重大快訊。"
                
                return data.dropna(), full_name, news_summary
            time.sleep(1.2)
        except Exception:
            time.sleep(1.2)
            continue
    return None, symbol, "獲取數據失敗"

# --- 3. AI 綜合分析與建議價邏輯 (要求4) ---
def perform_ai_analysis(df, precision, stock_name, news_txt):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 計算偏差值 (管理員設定)
    bias = (int(precision) - 55) / 100
    
    # A. 動能權重 (MACD)
    macd_factor = 1.02 if last['Hist'] > prev['Hist'] else 0.98
    # B. 量能權重
    vol_factor = 1.01 if last['Volume'] > last['V_MA5'] else 0.99
    # C. 超買賣權重 (KDJ)
    kdj_factor = 1.03 if last['K'] < 25 else (0.97 if last['K'] > 75 else 1.0)
    
    # 綜合修正係數
    total_mod = macd_factor * vol_factor * kdj_factor + bias
    
    # 計算買賣建議價 (基於月線支撐與壓力回歸)
    buy_price = last['MA20'] * 0.965 * total_mod
    sell_price = last['MA20'] * 1.045 * total_mod
    
    # 市場診斷評分
    base_score = 50
    if last['Close'] > last['MA20']: base_score += 10
    if last['Hist'] > 0: base_score += 10
    if last['K'] < 50: base_score += 5
    
    return {
        "name": stock_name,
        "buy": buy_price,
        "sell": sell_price,
        "score": base_score,
        "news": news_txt
    }

# --- 4. 儀表板繪製層 (要求1, 3, 6) ---
def show_ultimate_dashboard(symbol, p_days, precision):
    df, full_name, news_summary = fetch_comprehensive_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取股票 '{symbol}'，請確認網路或代碼。")
        return

    # 執行 AI 分析
    ai = perform_ai_analysis(df, precision, full_name, news_summary)
    
    # 標題顯示完整名稱 (要求6)
    st.title(f"📊 {ai['name']} ({symbol})")
    
    # 頂部 AI 區塊 (要求4)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='diag-box'>🟢 AI 建議買入價<br><span class='price-tag' style='color:#00FF41'>{ai['buy']:.2f}</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='diag-box'>🔴 AI 建議賣出價<br><span class='price-tag' style='color:#FF3131'>{ai['sell']:.2f}</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='diag-box'>⚖️ AI 綜合評分<br><span class='price-tag' style='color:#00F5FF'>{ai['score']}</span></div>", unsafe_allow_html=True)

    # 四層結構圖表 (要求1: 固定日線, 要求3: 加入KDJ)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.45, 0.15, 0.15, 0.25], vertical_spacing=0.03)
    
    p_df = df.tail(65) # 固定觀察區間
    
    # 1. K線層
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    
    # 2. 成交量層 (漲綠跌紅)
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量', marker_color=v_colors), 2, 1)
    
    # 3. MACD 層
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2)), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], name='訊號', line=dict(color='#FFFF00', width=1)), 3, 1)
    
    # 4. KDJ 層 (要求3: 綠線加粗)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(綠加粗)', line=dict(color='#00FF41', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J線', line=dict(color='#FF00FF', width=1.2)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)

    # AI 新聞重點摘要
    st.markdown("### 📰 AI 市場重點解析")
    st.markdown(f"""
        <div class='summary-card'>
            <p style='font-size:1.15rem; line-height:1.7;'>{ai['news']}</p>
            <hr style='border: 0.5px solid #30363D;'>
            <p style='color:#00F5FF; font-weight:bold;'>💡 AI 操作指引：依據 KDJ 超賣程度與 MACD 能量柱變化，建議於 {ai['buy']:.2f} 附近布局，目標價看至 {ai['sell']:.2f}。</p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. 主程式與持久化登入 (要求5) ---
def main():
    # 持久化登入鎖 (要求5: 30分鐘長效)
    @st.cache_resource(ttl=1800)
    def persistent_auth():
        return {"user": None}
    
    vault = persistent_auth()
    
    if 'user' not in st.session_state:
        st.session_state.user = vault["user"]

    # 數據庫連線
    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except Exception:
        st.error("⚠️ 雲端數據庫同步中，請稍候..."); return

    # 讀取設定
    s_records = ws_s.get_all_records()
    sd = {i['setting_name']: i['value'] for i in s_records}
    cp = int(sd.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🚀 StockAI 安全登入")
        u_in = st.text_input("帳號")
        p_in = st.text_input("密碼", type="password")
        if st.button("啟動終端", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u_in) & (udf['password'].astype(str)==p_in)].empty:
                st.session_state.user = vault["user"] = u_in
                st.rerun()
            else: st.error("驗證失敗：帳號或密碼錯誤")
    else:
        # 控制面板 (要求1: 取消單位, 要求2: 標註最大30)
        with st.expander("⚙️ 終端管理面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的選股清單", user_stocks if user_stocks else ["2330.TW"])
                if st.button("🚪 安全登出系統"):
                    st.session_state.user = vault["user"] = None
                    st.rerun()
            with m2:
                # 要求2
                p_days = st.number_input("AI 預測天數 (最大值30)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("管理員靈敏度調校", 0, 100, cp)
                    if st.button("💾 同步雲端設定"):
                        ws_s.update_cell(2, 2, str(new_p))
                        st.rerun()
        
        show_ultimate_dashboard(target, p_days, cp)

if __name__ == "__main__":
    main()
