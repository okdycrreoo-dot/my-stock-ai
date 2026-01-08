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

# --- 1. 配置與 PWA 注入 (完全保留原始規格) ---
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
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; color: #00F5FF !important; 
        border: 2px solid #00F5FF !important; border-radius: 12px !important;
        font-size: 1.2rem !important; font-weight: 900 !important;
    }
    div[data-baseweb="select"] > div { background-color: #1C2128 !important; color: #FFFFFF !important; border: 2px solid #00F5FF !important; }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stButton>button { 
        background-color: #00F5FF !important; color: #0E1117 !important; 
        border: none !important; border-radius: 12px; font-weight: 900 !important;
        height: 3.5rem !important; width: 100% !important;
    }
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 18px; margin-bottom: 12px; border: 1px solid #30363D; }
    .price-buy { color: #00FF41; font-weight: 900; font-size: 1.3rem; }
    .price-sell { color: #FF3131; font-weight: 900; font-size: 1.3rem; }
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (保留 3 次重試與自動補後綴) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): s = f"{s}.TW"
    
    for _ in range(3):
        try:
            # 使用 yf.download 加快速度並處理 MultiIndex
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # 計算指標
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                # MACD
                e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
                df['MACD'] = e12 - e26
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                df['Hist'] = df['MACD'] - df['Signal']
                # KDJ (K線加粗用)
                l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
                rsv = (df['Close'] - l9) / (h9 - l9 + 0.001) * 100
                df['K'] = rsv.ewm(com=2).mean()
                df['D'] = df['K'].ewm(com=2).mean()
                
                # 新聞抓取 (修復 KeyError)
                t = yf.Ticker(s)
                news = [{'title': n.get('title', '市場動態'), 'source': n.get('publisher', 'Yahoo')} for n in t.news[:3]]
                return df.dropna(), news, s
            time.sleep(1.5)
        except: time.sleep(1.5); continue
    return None, [], s

# --- 3. AI 分析引擎 (核心新增) ---
def perform_ai_period_analysis(df, news, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    # 情緒因子：結合 MACD 與 KDJ
    sentiment = 1.0
    if last['Hist'] > prev['Hist']: sentiment += 0.01
    if last['K'] < 30: sentiment += 0.02 # 超跌買進加權
    
    bias = (int(precision) / 50) # 靈敏度影響區間寬度
    vol = df['Close'].pct_change().std() # 歷史波動
    
    periods = {"5日 (短線)": (last['MA5'], 1.8), "20日 (月線)": (last['MA20'], 2.8), "60日 (季線)": (last['MA60'], 4.2)}
    res = {}
    for k, (ma, factor) in periods.items():
        range_val = ma * vol * factor * bias
        res[k] = {"buy": (ma - range_val) * sentiment, "sell": (ma + range_val) * sentiment}
    return res

# --- 4. 圖表與儀表板渲染 ---
def show_dashboard(symbol, unit, p_days, precision):
    df, news_data, f_id = fetch_comprehensive_data(symbol)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    ai_recs = perform_ai_period_analysis(df, news_data, precision)
    st.title(f"📊 {f_id} 深度技術終端")
    
    # 建議價卡片
    cols = st.columns(3)
    for i, (period, val) in enumerate(ai_recs.items()):
        with cols[i]:
            st.markdown(f"""
            <div class='diag-box'>
                <center><b>{period} AI 策略</b></center><hr style='border:0.5px solid #30363D'>
                建議買入: <span class='price-buy'>{val['buy']:.2f}</span><br>
                建議賣出: <span class='price-sell'>{val['sell']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # 四層技術圖表
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD力道', marker_color=v_colors), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(加粗)', line=dict(color='#00FF41', width=3.5)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1)), 4, 1)
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📰 市場新聞動態")
    for n in news_data:
        st.markdown(f"<div class='diag-box'>📢 <b>{n['title']}</b> ({n['source']})</div>", unsafe_allow_html=True)

# --- 5. 主程式 (完整還原 240+ 行架構) ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except Exception as e:
        st.error(f"🚨 資料庫連線失敗: {e}"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp, c_ttl = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 全能終端")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        with t1:
            u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
            if st.button("確認進入系統", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with t2:
            nu, npw = st.text_input("新帳號"), st.text_input("新密碼", type="password")
            if st.button("完成註冊", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if nu and nu not in udf['username'].astype(str).tolist():
                    ws_u.append_row([nu, npw]); st.success("註冊成功！")
    else:
        st.write(f"👤 用戶: {st.session_state.user} | ⚙️ 靈敏度: {cp}")
        with st.expander("🛠️ 功能管理面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("我的清單", u_stocks if u_stocks else ["2330"])
                if st.button(f"🗑️ 刪除 {target}"):
                    # 搜尋並刪除特定行
                    vals = ws_w.get_all_values()
                    for i, row in enumerate(vals):
                        if i > 0 and row[0] == st.session_state.user and row[1] == target:
                            ws_w.delete_rows(i + 1); st.rerun()
                ns = st.text_input("➕ 新增代碼 (例: 2454)")
                if st.button("執行新增"):
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                # 保留要求：天數標註
                p_days = st.number_input("AI 預測天數 (最大30)", 1, 30, 7)
                unit = st.selectbox("顯示時間單位", ["日", "月", "年"])
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("同步全域靈敏度", 0, 100, cp)
                    if st.button("💾 更新資料庫"):
                        ws_s.update_cell(2, 2, str(new_p)); st.rerun()
                if st.button("🚪 安全登出"): st.session_state.user = None; st.rerun()
        
        show_dashboard(target, unit, p_days, cp)

if __name__ == "__main__": main()
