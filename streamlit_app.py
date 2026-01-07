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

# --- 1. 頁面配置與高對比主題 ---
st.set_page_config(page_title="StockAI 專業管理終端", layout="wide")

# 強制修正深色高對比度 CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00F5FF; font-weight: bold; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    .stCaption { color: #8B949E; font-size: 0.85rem; }
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
    except Exception as e:
        st.error(f"Google API 連線失敗: {e}")
        return None

# --- 3. 全域設定讀取 ---
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

# --- 4. 核心 AI 預測邏輯 (權重擬合) ---
def run_ai_prediction(df, predict_days, precision):
    last_p = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    
    # 計算偏差權重
    rsi_bias = (50 - rsi) * 0.001
    trend_bias = (ma20 - last_p) / ma20 * 0.05
    user_bias = (int(precision) / 100) * 0.01
    
    pred_prices = []
    curr_p = last_p
    for i in range(1, predict_days + 1):
        noise = np.random.normal(0, 0.0015)
        change = 1 + (user_bias + rsi_bias + trend_bias + noise)
        curr_p *= change
        pred_prices.append(curr_p)
    return pred_prices

# --- 5. 智慧掃描與圖表繪製 ---
def show_analysis_dashboard(symbol, unit, p_days, precision):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty:
            st.warning(f"無法取得 {symbol} 的數據")
            return
        
        # 指標計算
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        pred_prices = run_ai_prediction(df, p_days, precision)
        
        # 顯示指標卡片
        last_p = df['Close'].iloc[-1]
        target_p = pred_prices[-1]
        pct = ((target_p - last_p)/last_p)*100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("當前收盤", f"{last_p:.2f}")
        c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
        c3.metric("預計漲跌", f"{pct:.2f}%", delta=f"{pct:.2f}%")

        # Plotly 子圖繪製
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        zoom = {"日": 40, "月": 250, "年": 750}[unit]
        p_df = df.tail(zoom)
        
        # K線與預測線
        fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='月線 MA20', line=dict(color='#00F5FF', width=2)), row=1, col=1)
        
        f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
        fig.add_trace(go.Scatter(x=f_dates, y=pred_prices, name='AI 預測路徑', line=dict(color='#FF4500', width=4, dash='dash')), row=1, col=1)
        
        # 交易量
        fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='交易量', marker_color='#30363D'), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=650, paper_bgcolor="#0E1117", plot_bgcolor="#161B22", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"系統錯誤: {e}")

# --- 6. 主程式邏輯 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    client = get_google_client()
    if not client: return
    
    # 讀取管理員全域設定
    settings_ws, settings = get_global_settings(client)
    curr_prec = settings.get('global_precision', 55)
    curr_ttl = int(settings.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級管理終端")
        tab_l, tab_r = st.tabs(["🔑 登入", "📝 註冊"])
        # (此處為標準 gspread 登入邏輯)
        with tab_l:
            with st.form("login"):
                u = st.text_input("帳號")
                p = st.text_input("密碼", type="password")
                if st.form_submit_button("登入系統", use_container_width=True):
                    user_df = pd.DataFrame(client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).get_all_records())
                    if not user_df[(user_df['username'].astype(str)==u) & (user_df['password'].astype(str)==p)].empty:
                        st.session_state.user = u
                        st.rerun()
                    else: st.error("帳密錯誤")
    else:
        # --- 登入後的頂部狀態欄 ---
        next_sync = st.session_state.last_sync + timedelta(minutes=curr_ttl)
        remain = next_sync - datetime.now()
        
        st.markdown(f"**使用者：{st.session_state.user}**")
        c_s1, c_s2 = st.columns([2,1])
        c_s1.caption(f"🕒 上次 API 同步：{st.session_state.last_sync.strftime('%H:%M:%S')}")
        c_s2.caption(f"⏳ 下次更新倒數：{max(0, int(remain.total_seconds() // 60))}分 {max(0, int(remain.total_seconds() % 60))}秒")
        st.markdown("---")

        # --- 側邊欄控制與權限管理 ---
        with st.sidebar:
            st.title("控制面板")
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員權限控制", expanded=True):
                    new_p = st.slider("設定全域靈敏度", 0, 100, int(curr_prec))
                    new_t = st.select_slider("設定 API 快取分鐘", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("更新全域設定"):
                        settings_ws.update_cell(2, 2, str(new_p))
                        settings_ws.update_cell(3, 2, str(new_t))
                        st.session_state.last_sync = datetime.now() # 強制更新時間
                        st.success("同步成功")
                        time.sleep(1)
                        st.rerun()
            else:
                st.info(f"系統靈敏度：{curr_prec}%")
                st.info(f"API 快取：{curr_ttl} min")

            # 股票選擇與管理
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            ws_watch = client.open_by_url(url).worksheet("watchlist")
            all_watch = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = all_watch[all_watch['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_watch.empty else []
            
            target = st.selectbox("我的自選股", user_stocks if user_stocks else ["2330.TW"])
            unit = st.selectbox("圖表單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測延伸天數", 1, 30, 7)
            
            if st.button("🗑️ 刪除目前股票"):
                cells = ws_watch.findall(st.session_state.user)
                for c in cells:
                    if ws_watch.row_values(c.row)[1] == target:
                        ws_watch.delete_rows(c.row); st.rerun()
            
            new_s = st.text_input("新增代碼 (限30筆)").strip().upper()
            if st.button("➕ 新增"):
                if len(user_stocks) < 30 and new_s and new_s not in user_stocks:
                    ws_watch.append_row([st.session_state.user, new_s]); st.rerun()

            if st.button("🚪 登出"):
                st.session_state.user = None; st.rerun()

        # --- 主畫面顯示 ---
        show_analysis_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
