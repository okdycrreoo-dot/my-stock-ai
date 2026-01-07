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

# --- 1. 初始化與主題配置 ---
st.set_page_config(page_title="StockAI 高級技術終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00F5FF; font-weight: bold; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    .stCaption { color: #8B949E; font-size: 0.8rem; }
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
    except:
        return None

# --- 3. 數據抓取優化 (加入備援與格式修復) ---
@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    for _ in range(3): # 失敗重試 3 次
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True)
            if not data.empty and len(data) > 30:
                # 修復 MultiIndex 問題
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 計算技術指標
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                std = data['Close'].rolling(20).std()
                data['BB_up'] = data['MA20'] + (std * 2)
                data['BB_low'] = data['MA20'] - (std * 2)
                
                # 自動判斷支撐壓力位
                recent = data.tail(60)
                data['Support'] = recent['Low'].min()
                data['Resistance'] = recent['High'].max()
                
                return data.dropna()
            time.sleep(1)
        except:
            continue
    return None

# --- 4. 繪圖與 AI 預測引擎 ---
def show_professional_chart(symbol, unit, p_days, precision):
    df = fetch_stock_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取股票代碼 '{symbol}'。請確認格式是否正確 (如: 2330.TW)")
        return

    # AI 向量化運算 (加速版)
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    trend = (int(precision) - 55) / 500
    pred_ratios = np.cumprod(1 + trend + noise)
    pred_prices = last_p * pred_ratios

    # 指標卡片
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
    c3.metric("預期漲跌", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # 專業繪圖 (使用 Scattergl 加速)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    zoom = {"日": 45, "月": 200, "年": 550}[unit]
    p_df = df.tail(zoom)
    
    # 主圖
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), row=1, col=1)
    
    # 布林通道
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_up'], name='布林上', line=dict(width=0)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['BB_low'], name='布林下', fill='tonexty', fillcolor='rgba(128,128,128,0.1)', line=dict(width=0)), row=1, col=1)
    
    # 支撐壓力線
    fig.add_hline(y=p_df['Support'].iloc[-1], line_dash="dash", line_color="green", annotation_text="支撐", row=1, col=1)
    fig.add_hline(y=p_df['Resistance'].iloc[-1], line_dash="dash", line_color="red", annotation_text="壓力", row=1, col=1)

    # AI 預測路徑
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=3, dash='dashdot')), row=1, col=1)

    # 交易量
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color='#30363D'), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    client = get_google_client()
    if not client:
        st.error("🚨 系統連線環境異常，請檢查 Secrets。")
        return

    # 嘗試讀取設定，失敗則使用安全預設
    try:
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_settings = sh.worksheet("settings")
        s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
        curr_prec = int(s_data.get('global_precision', 55))
        curr_ttl = int(s_data.get('api_ttl_min', 5))
    except:
        curr_prec, curr_ttl = 55, 5

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級技術終端")
        tab_l, tab_r = st.tabs(["🔑 登入", "📝 註冊"])
        with tab_l:
            with st.form("login"):
                u = st.text_input("帳號")
                p = st.text_input("密碼", type="password")
                if st.form_submit_button("登入系統", use_container_width=True):
                    try:
                        user_df = pd.DataFrame(sh.worksheet("users").get_all_records())
                        if not user_df[(user_df['username'].astype(str)==u) & (user_df['password'].astype(str)==p)].empty:
                            st.session_state.user = u
                            st.rerun()
                        else: st.error("帳號或密碼錯誤")
                    except: st.error("資料庫讀取失敗")
    else:
        # 頂部狀態列
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.caption(f"👤 {st.session_state.user} | 🕒 上次同步: {st.session_state.last_sync.strftime('%H:%M:%S')} | ⏳ 剩餘快取: {max(0, int(remain.total_seconds()))}s")
        st.markdown("---")

        with st.sidebar:
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員權限控制", expanded=True):
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("API 快取 (分)", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("同步設定並強制重置快取"):
                        ws_settings.update_cell(2, 2, str(new_p))
                        ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear()
                        st.session_state.last_sync = datetime.now()
                        st.rerun()
            
            # 股票清單管理
            ws_watch = sh.worksheet("watchlist")
            user_stocks = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = user_stocks[user_stocks['username'] == st.session_state.user]['stock_symbol'].tolist() if not user_stocks.empty else []
            
            target = st.selectbox("自選股清單", user_stocks if user_stocks else ["2330.TW"])
            unit = st.selectbox("時間單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測延伸天數", 1, 30, 7)
            
            if st.button("➕ 新增代碼"):
                st.write("請在下方輸入後點擊新增")
            new_s = st.text_input("代碼 (如: AAPL)").strip().upper()
            if st.button("確認新增"):
                if new_s and new_s not in user_stocks:
                    ws_watch.append_row([st.session_state.user, new_s]); st.rerun()
            
            if st.button("🚪 登出系統"):
                st.session_state.user = None; st.rerun()

        show_professional_chart(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
