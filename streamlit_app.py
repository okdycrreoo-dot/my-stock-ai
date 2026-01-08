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

# --- 1. 配置與 PWA 強制圖示注入 (維持原樣) ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

st.markdown("""
    <link rel="manifest" href="./manifest.json?v=1.1">
    <link rel="apple-touch-icon" href="./icon.png?v=1.1">
    <link rel="icon" type="image/png" href="./icon.png?v=1.1">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', function() {
        navigator.serviceWorker.register('./sw.js?v=1.1');
      });
    }
    </script>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 600 !important; }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #00F5FF !important;
        border-radius: 12px !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
    }
    .streamlit-expanderHeader svg { fill: #00F5FF !important; }
    div[data-baseweb="select"] > div { 
        background-color: #1C2128 !important; 
        color: #FFFFFF !important; 
        border: 2px solid #00F5FF !important; 
        border-radius: 10px !important;
    }
    div[role="listbox"] div { color: #000000 !important; } 
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    div[data-baseweb="input"] > div { background-color: #1C2128 !important; }
    .stSlider [data-testid="stWidgetLabel"] p { color: #00F5FF !important; }
    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border: none !important;
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important;
        width: 100% !important;
    }
    [data-testid="stMetricValue"] { color: #00F5FF !important; font-weight: bold; }
    .stMetric { background-color: #1C2128; border: 2px solid #30363D; border-radius: 15px; padding: 20px; }
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 18px; margin-bottom: 12px; border: 1px solid #30363D; }
    .price-buy { color: #00FF41; font-weight: 900; font-size: 1.2rem; }
    .price-sell { color: #FF3131; font-weight: 900; font-size: 1.2rem; }
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (優化 MultiIndex 解析與指標計算) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for _ in range(3):
        try:
            # 確保台股代碼格式
            s = symbol.upper()
            if not (s.endswith(".TW") or s.endswith(".TWO")):
                s = f"{s}.TW"
            
            data = yf.download(s, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 基礎均線
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                data['MA60'] = data['Close'].rolling(60).mean()
                
                # MACD
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                
                # KDJ (9,3,3)
                low_list = data['Low'].rolling(9, min_periods=9).min()
                high_list = data['High'].rolling(9, min_periods=9).max()
                rsv = (data['Close'] - low_list) / (high_list - low_list) * 100
                data['K'] = rsv.ewm(com=2, adjust=False).mean()
                data['D'] = data['K'].ewm(com=2, adjust=False).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                
                # 獲取新聞 (情緒因子)
                ticker = yf.Ticker(s)
                news = ticker.news[:3]
                
                return data.dropna(), news, s
            time.sleep(1.5)
        except:
            time.sleep(1.5); continue
    return None, [], symbol

# --- 3. AI 多週期價格建議邏輯 ---
def perform_ai_price_analysis(df, news, precision):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 基礎情緒得分 (根據 KDJ 與 MACD)
    sentiment_score = 0
    if last['K'] < 20: sentiment_score += 10  # 超賣區
    if last['K'] > 80: sentiment_score -= 10  # 超買區
    if last['Hist'] > prev['Hist']: sentiment_score += 5 # 動能增加
    
    # 管理員靈敏度權重 (0-100 映射為 0.8 - 1.2)
    sensitivity = (int(precision) / 50) 
    volatility = df['Close'].pct_change().std() # 波動率
    
    # 計算各週期支撐與壓力 (建議買賣價)
    def calc_levels(ma_val, days):
        # 波動擴張因子：天數愈長，區間愈大
        range_factor = volatility * np.sqrt(days) * sensitivity
        buy_price = ma_val * (1 - range_factor)
        sell_price = ma_val * (1 + range_factor)
        # 根據情緒微調
        adj = 1 + (sentiment_score / 500)
        return buy_price * adj, sell_price * adj

    res = {
        "5日 (短線)": calc_levels(last['MA5'], 5),
        "20日 (月線)": calc_levels(last['MA20'], 20),
        "60日 (季線)": calc_levels(last['MA60'], 60)
    }
    return res

# --- 4. 繪圖展示與 AI 建議面板 ---
def show_ultimate_dashboard(symbol, unit, p_days, precision):
    df, news_data, final_symbol = fetch_comprehensive_data(symbol)
    if df is None: st.error(f"❌ 無法讀取股票代碼 '{symbol}'"); return

    # 執行分析
    ai_levels = perform_ai_price_analysis(df, news_data, precision)
    last_p = float(df['Close'].iloc[-1])

    # 頂部指標
    c1, c2, c3 = st.columns(3)
    c1.metric("當前收盤價", f"{last_p:.2f}")
    c2.metric("MACD 趨勢", "🔥 偏多" if df['Hist'].iloc[-1] > 0 else "❄️ 偏空")
    c3.metric("KDJ 狀態", f"K:{df['K'].iloc[-1]:.1f}")

    st.markdown("### 🤖 AI 多週期操作建議價")
    rows = st.columns(3)
    for i, (period, prices) in enumerate(ai_levels.items()):
        with rows[i]:
            st.markdown(f"""
            <div class='diag-box'>
                <center><b>{period}</b></center>
                <hr style='border: 0.5px solid #30363D;'>
                建議買入: <span class='price-buy'>{prices[0]:.2f}</span><br>
                建議賣出: <span class='price-sell'>{prices[1]:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # 繪製四層技術圖表
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    
    zoom = {"日": 60, "月": 180, "年": 500}[unit]
    p_df = df.tail(zoom)
    
    # 1. 主圖: K線 + 三均線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA60'], name='MA60', line=dict(color='#FF00FF', width=1)), 1, 1)
    
    # 2. 成交量
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    
    # 3. MACD
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD Hist', marker_color=v_colors), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#FFFFFF', width=1)), 3, 1)
    
    # 4. KDJ (K線加粗)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K線(加粗)', line=dict(color='#00FF41', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J線', line=dict(color='#FF00FF', width=1)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=1000, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 市場資訊面板
    st.markdown("### 📰 相關市場動態分析")
    if news_data:
        for n in news_data:
            st.markdown(f"<div class='diag-box'>📌 <b>{n['title']}</b><br><small>來源: {n.get('publisher','Yahoo Finance')}</small></div>", unsafe_allow_html=True)
    else:
        st.info("暫時無相關新聞資訊")

# --- 5. 主程式 (結構維持不變) ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_user, ws_watch, ws_settings = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except Exception as e:
        st.error(f"🚨 數據庫連線異常: {e}"); return

    s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
    curr_prec, curr_ttl = int(s_data.get('global_precision', 55)), int(s_data.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 全能技術終端")
        t1, t2 = st.tabs(["🔑 系統登入", "📝 快速註冊"])
        with t1:
            u, p = st.text_input("帳號", key="login_u"), st.text_input("密碼", type="password", key="login_p")
            if st.button("確認登入", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with t2:
            nu, npw = st.text_input("新帳號", key="reg_u"), st.text_input("新密碼", type="password", key="reg_p")
            if st.button("註冊帳號", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if nu and nu not in udf['username'].astype(str).tolist():
                    ws_user.append_row([nu, npw]); st.success("註冊成功！")
                else: st.error("帳號已存在")
    else:
        # 已登入介面
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.markdown(f"👤 **{st.session_state.user}** | 🕒 刷新倒數: **{max(0, int(remain.total_seconds()))}s**")
        
        with st.expander("⚙️ 終端功能面板", expanded=False):
            m1, m2 = st.columns([1, 1])
            with m1:
                st.subheader("📋 清單管理")
                all_w = pd.DataFrame(ws_watch.get_all_records())
                user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的選股清單", user_stocks if user_stocks else ["2330"], key="stock_sel")
                ns = st.text_input("➕ 新增代碼", key="add_stock")
                if st.button("確認新增"):
                    if ns and ns.upper() not in user_stocks:
                        ws_watch.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                st.subheader("🛠️ 參數設定")
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("快取連線時間", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("💾 同步資料庫"):
                        ws_settings.update_cell(2, 2, str(new_p))
                        ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear(); st.session_state.last_sync = datetime.now(); st.rerun()
                unit = st.selectbox("時間單位", ["日", "月", "年"], key="time_unit")
                if st.button("🚪 登出系統"): st.session_state.user = None; st.rerun()

        # 渲染主圖表與建議
        show_ultimate_dashboard(target, unit, 7, curr_prec)

if __name__ == "__main__":
    main()
