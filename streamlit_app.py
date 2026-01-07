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

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 高級管理終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00F5FF; font-weight: bold; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安全連線核心 ---
@st.cache_resource
def get_google_client():
    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except: return None

# --- 3. 數據抓取備援機制 (修正版) ---
@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    for _ in range(3):
        try:
            # 增加 auto_adjust 並強制處理數據
            data = yf.download(symbol, period="2y", interval="1d", 
                               progress=False, threads=False, auto_adjust=True)
            if not data.empty and len(data) > 20:
                # 關鍵修正：確保移除 MultiIndex 並處理 NaN
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data.dropna()
            time.sleep(1)
        except:
            continue
    return None

# --- 4. 管理與 AI 邏輯 ---
def get_global_settings(client):
    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    sh = client.open_by_url(url)
    try:
        ws = sh.worksheet("settings")
    except:
        ws = sh.add_worksheet(title="settings", rows="10", cols="2")
        ws.append_row(["setting_name", "value"])
        ws.append_row(["global_precision", "55"])
        ws.append_row(["api_ttl_min", "5"])
    
    data = ws.get_all_records()
    settings = {item['setting_name']: item['value'] for item in data}
    return ws, settings

def run_ai_prediction(df, predict_days, precision):
    # 強制轉換為純數字防止 TypeError
    last_p = float(df['Close'].iloc[-1])
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    
    rsi_bias = (50 - float(rsi)) * 0.001
    trend_bias = (float(ma20) - last_p) / float(ma20) * 0.05
    user_bias = (int(precision) / 100) * 0.01
    
    pred_prices = []
    curr_p = last_p
    for i in range(1, predict_days + 1):
        noise = np.random.normal(0, 0.0015)
        curr_p *= (1 + (user_bias + rsi_bias + trend_bias + noise))
        pred_prices.append(float(curr_p))
    return pred_prices

# --- 5. 主畫面繪圖 (修復格式化錯誤) ---
def show_analysis_dashboard(symbol, unit, p_days, precision):
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        st.error(f"❌ 無法取得代碼 '{symbol}' 的有效數據。請檢查格式（如: 2330.TW）。")
        return
    
    try:
        # 強制轉換最後一個收盤價為 float
        last_p = float(df['Close'].iloc[-1])
        pred_prices = run_ai_prediction(df, p_days, precision)
        target_p = float(pred_prices[-1])
        pct = ((target_p - last_p)/last_p)*100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("當前收盤", f"{last_p:.2f}")
        c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
        c3.metric("預計漲跌", f"{pct:.2f}%", delta=f"{pct:.2f}%")

        # 繪製圖表 (同前，略...)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        zoom = {"日": 40, "月": 250, "年": 750}[unit]
        p_df = df.tail(zoom)
        fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
        f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
        fig.add_trace(go.Scatter(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4, dash='dash')), row=1, col=1)
        fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='交易量', marker_color='#30363D'), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"數據解析錯誤: {e}")

# --- 6. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    client = get_google_client()
    if not client: return
    
    settings_ws, settings = get_global_settings(client)
    curr_prec = settings.get('global_precision', 55)
    curr_ttl = int(settings.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級管理終端")
        t_l, t_r = st.tabs(["🔑 登入", "📝 註冊"])
        with t_l:
            with st.form("login"):
                u = st.text_input("帳號")
                p = st.text_input("密碼", type="password")
                if st.form_submit_button("登入"):
                    user_df = pd.DataFrame(client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).get_all_records())
                    if not user_df[(user_df['username'].astype(str)==u) & (user_df['password'].astype(str)==p)].empty:
                        st.session_state.user = u
                        st.rerun()
    else:
        # 顯示同步狀態
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.caption(f"👤 使用者：{st.session_state.user} | ⏳ 剩餘快取：{max(0, int(remain.total_seconds()))} 秒")
        
        with st.sidebar:
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員控制台"):
                    new_p = st.slider("靈敏度", 0, 100, int(curr_prec))
                    new_t = st.select_slider("快取分鐘", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("同步全域設定"):
                        settings_ws.update_cell(2, 2, str(new_p))
                        settings_ws.update_cell(3, 2, str(new_t))
                        st.cache_data.clear()
                        st.session_state.last_sync = datetime.now()
                        st.rerun()
            
            # 股票選擇邏輯 (同前，略...)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            ws_watch = client.open_by_url(url).worksheet("watchlist")
            all_watch = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = all_watch[all_watch['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_watch.empty else []
            target = st.selectbox("我的自選股", user_stocks if user_stocks else ["2330.TW"])
            unit = st.selectbox("圖表單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測延伸(天)", 1, 30, 7)
            
            if st.button("🚪 登出"):
                st.session_state.user = None; st.rerun()

        show_analysis_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
