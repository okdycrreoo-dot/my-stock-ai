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

# --- 1. 配置與專業視覺優化 (完整保留 CSS) ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 600 !important; }
    div[data-baseweb="select"] > div { color: #FFFFFF !important; background-color: #1C2128 !important; }
    section[data-testid="stSidebar"] { background-color: #11151C !important; border-right: 1px solid #30363D; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #00F5FF !important; }
    [data-testid="stMetricValue"] { color: #00F5FF !important; font-weight: bold; font-size: 2.2rem !important; }
    [data-testid="stMetricLabel"] { color: #CCCCCC !important; font-size: 1.1rem !important; }
    .stMetric { background-color: #1C2128; border: 2px solid #30363D; border-radius: 15px; padding: 20px; }
    .streamlit-expanderHeader { background-color: #232931 !important; color: #00F5FF !important; font-size: 1.1rem !important; }
    .stButton>button { background-color: #2D333B !important; color: #FFFFFF !important; border: 1px solid #444C56 !important; border-radius: 8px; font-weight: bold !important; }
    .stButton>button:hover { border-color: #00F5FF !important; color: #00F5FF !important; }
    /* 診斷卡片樣式 */
    .diag-box { background-color: #161B22; border: 1px solid #30363D; border-left: 5px solid #00F5FF; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (完整 3 次重試與指標計算) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for _ in range(3):
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True, repair=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                return data.dropna()
            time.sleep(1.5)
        except:
            time.sleep(1.5); continue
    return None

# --- 3. 新增：AI 多因子診斷與情感分析邏輯 ---
def get_ai_diagnosis(df, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 50
    reasons = []
    
    # 技術因子
    if last['Close'] > last['MA20']:
        score += 15; reasons.append("🟢 股價站於月線 (MA20) 之上，趨勢偏多。")
    else:
        score -= 10; reasons.append("🔴 股價跌破月線，需注意回檔風險。")
        
    if last['Hist'] > prev['Hist']:
        score += 10; reasons.append("🔥 MACD 柱狀體放大，多頭動能轉強。")
    
    # 情感分析模擬
    news_items = [
        {"title": "產業景氣回溫，分析師調高投資評等", "sentiment": "利多", "weight": 10},
        {"title": "短期面臨匯率波動與通膨壓力", "sentiment": "中性", "weight": -3}
    ]
    news_total = sum(item['weight'] for item in news_items)
    
    final_score = max(0, min(100, score + news_total + (int(precision)-55)))
    return int(final_score), reasons, news_items

# --- 4. 視覺強化繪圖引擎 (完整恢復原本的所有細節) ---
def show_ultimate_dashboard(symbol, unit, p_days, precision):
    df = fetch_comprehensive_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取 '{symbol}'，請檢查代碼或稍後再試。")
        return

    # AI 預測邏輯 (原本的 Monte Carlo)
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    trend = (int(precision) - 55) / 500
    pred_prices = last_p * np.cumprod(1 + trend + noise)

    # 診斷計算
    score, reasons, news = get_ai_diagnosis(df, precision)

    # 頂部卡片 (救回原本的 Metric 配置)
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
    c3.metric("預期回報", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # AI 診斷與新聞區域 (新功能整合)
    st.divider()
    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        st.subheader("💡 AI 因子分析")
        for r in reasons: st.write(r)
        st.markdown(f"**AI 綜合評分: `{score}` / 100**")
    with d_col2:
        st.subheader("📰 新聞情感分析")
        for n in news:
            st.markdown(f"<div class='diag-box'><b>[{n['sentiment']}]</b> {n['title']}</div>", unsafe_allow_html=True)

    # 圖表配置 (完全保留原本細節)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.3], vertical_spacing=0.04)
    zoom = {"日": 45, "月": 180, "年": 550}[unit]
    p_df = df.tail(zoom)
    
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線', increasing_line_color='#00FF41', decreasing_line_color='#FF3131'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=2.5)), row=1, col=1)
    
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4.5, dash='dashdot')), row=1, col=1)
    
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors, opacity=0.8), row=2, col=1)
    
    h_colors = ['#FF3131' if v < 0 else '#00FF41' for v in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD力道', marker_color=h_colors), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.info(f"📍 **總結說明**：目前 {symbol} 在技術面上表現 {'偏多' if score > 50 else '偏空'}。AI 綜合多個因子（如月線支撐、MACD動能與情感分析）後給出 {score} 分。建議搭配下方 K 線圖確認具體進場位。")

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_user, ws_watch, ws_settings = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except:
        st.error("🚨 數據庫連線異常。"); return

    s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
    curr_prec, curr_ttl = int(s_data.get('global_precision', 55)), int(s_data.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級技術終端")
        t1, t2 = st.tabs(["🔑 登入系統", "📝 註冊帳號"])
        with t1:
            u, p = st.text_input("帳號", key="login_u"), st.text_input("密碼", type="password", key="login_p")
            if st.button("確認登入", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with t2:
            nu, npw = st.text_input("新帳號"), st.text_input("新密碼", type="password")
            if st.button("註冊帳號", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if nu and nu not in udf['username'].astype(str).tolist():
                    ws_user.append_row([nu, npw]); st.success("註冊成功！請登入。")
                else: st.error("帳號已存在或為空")
    else:
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.caption(f"👤 {st.session_state.user} | 🕒 刷新倒數: {max(0, int(remain.total_seconds()))}s")

        with st.sidebar:
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員控制台", expanded=True):
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("快取分鐘", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("💾 同步設定"):
                        ws_settings.update_cell(2, 2, str(new_p)); ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear(); st.session_state.last_sync = datetime.now(); st.rerun()
            
            st.subheader("📋 清單管理")
            all_w = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
            target = st.selectbox("我的選股", user_stocks if user_stocks else ["2330.TW"])
            
            if user_stocks and st.button(f"🗑️ 刪除 {target}", use_container_width=True):
                rows = ws_watch.get_all_values()
                for i, row in enumerate(rows):
                    if i > 0 and row[0] == st.session_state.user and row[1] == target:
                        ws_watch.delete_rows(i + 1); st.rerun()
            
            st.divider()
            ns = st.text_input("➕ 新增代碼").strip().upper()
            if st.button("確認新增", use_container_width=True):
                if ns and ns not in user_stocks:
                    ws_watch.append_row([st.session_state.user, ns]); st.rerun()
            
            st.divider()
            unit = st.selectbox("時間單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測天數", 1, 30, 7)
            if st.button("🚪 登出", use_container_width=True): st.session_state.user = None; st.rerun()

        show_ultimate_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
