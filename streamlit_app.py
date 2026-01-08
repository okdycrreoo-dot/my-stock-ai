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

# --- 1. 配置與專業視覺優化 ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

st.markdown("""
    <style>
    /* 全域背景與文字基礎顏色 */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* 強制顯示所有標籤與文字 (解決看不清楚的問題) */
    label, p, span, .stMarkdown { color: #FFFFFF !important; font-weight: 500; }
    
    /* 儀表板卡片視覺強化 */
    [data-testid="stMetricValue"] { color: #00F5FF !important; font-weight: bold; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #AAAAAA !important; font-size: 1rem !important; }
    .stMetric { 
        background-color: #1C2128; 
        border: 1px solid #30363D; 
        border-radius: 12px; 
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* 登入與註冊頁面 Input 框強化 */
    .stTextInput input {
        background-color: #161B22 !important;
        color: white !important;
        border: 1px solid #30363D !important;
    }
    
    /* 側邊欄文字強化 */
    section[data-testid="stSidebar"] { background-color: #161B22; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #E0E0E0 !important; }

    /* Tab 顏色與對比度 */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        font-size: 18px; 
        color: #888888 !important; 
    }
    .stTabs [aria-selected="true"] { color: #00F5FF !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (含 3 次重試與所有指標) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for _ in range(3):
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True, repair=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 計算指標
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                std = data['Close'].rolling(20).std()
                data['BB_up'] = data['MA20'] + (std * 2)
                data['BB_low'] = data['MA20'] - (std * 2)
                
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                
                recent = data.tail(60)
                data['Support'] = recent['Low'].min()
                data['Resistance'] = recent['High'].max()
                return data.dropna()
            time.sleep(1.5)
        except:
            time.sleep(1.5); continue
    return None

# --- 3. 視覺強化繪圖引擎 (修正圖例顏色) ---
def show_ultimate_dashboard(symbol, unit, p_days, precision):
    df = fetch_comprehensive_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取 '{symbol}'，請重新同步。")
        return

    # AI 預測邏輯
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    trend = (int(precision) - 55) / 500
    pred_prices = last_p * np.cumprod(1 + trend + noise)

    # 頂部卡片
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
    c3.metric("預期回報", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # 圖表配置
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.55, 0.15, 0.3], vertical_spacing=0.04)
    zoom = {"日": 45, "月": 180, "年": 550}[unit]
    p_df = df.tail(zoom)
    
    # 1. K線 (紅白對比)
    fig.add_trace(go.Candlestick(
        x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], 
        name='K線', increasing_line_color='#00FF41', decreasing_line_color='#FF3131'
    ), row=1, col=1)

    # 線條
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=2.5)), row=1, col=1)
    
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4, dash='dashdot')), row=1, col=1)

    # 2. 成交量
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors, opacity=0.8), row=2, col=1)

    # 3. MACD
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MACD'], name='MACD', line=dict(color='#00F5FF', width=2)), row=3, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['Signal'], name='訊號線', line=dict(color='#FFD700', width=2)), row=3, col=1)
    h_colors = ['#FF3131' if v < 0 else '#00FF41' for v in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='Hist', marker_color=h_colors), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, 
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 4. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_user = sh.worksheet("users")
        ws_watch = sh.worksheet("watchlist")
        ws_settings = sh.worksheet("settings")
    except:
        st.error("🚨 系統連線異常。")
        return

    # 管理員統一設定 (okdycrreoo 控制)
    try:
        s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
        curr_prec = int(s_data.get('global_precision', 55))
        curr_ttl = int(s_data.get('api_ttl_min', 5))
    except:
        curr_prec, curr_ttl = 55, 5

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級技術終端")
        tab_login, tab_reg = st.tabs(["🔑 登入系統", "📝 註冊帳號"])
        with tab_login:
            u = st.text_input("帳號", key="login_u")
            p = st.text_input("密碼", type="password", key="login_p")
            if st.button("確認登入", use_container_width=True):
                user_df = pd.DataFrame(ws_user.get_all_records())
                if not user_df[(user_df['username'].astype(str)==u) & (user_df['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with tab_reg:
            new_u = st.text_input("設定帳號", key="reg_u")
            new_p = st.text_input("設定密碼", type="password", key="reg_p")
            if st.button("確認註冊", use_container_width=True):
                user_df = pd.DataFrame(ws_user.get_all_records())
                if new_u in user_df['username'].astype(str).values: st.warning("帳號已存在")
                elif new_u and new_p:
                    ws_user.append_row([new_u, new_p]); st.success("註冊成功！")
    else:
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.caption(f"👤 {st.session_state.user} | 🕒 刷新倒數: {max(0, int(remain.total_seconds()))}s")

        with st.sidebar:
            if st.session_state.user == "okdycrreoo":
                with st.expander("🛠️ 管理員控制台 (okdycrreoo)", expanded=True):
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("快取分鐘", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("同步至資料庫"):
                        ws_settings.update_cell(2, 2, str(new_p))
                        ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear(); st.session_state.last_sync = datetime.now(); st.rerun()
            
            st.subheader("📋 清單管理")
            all_w = pd.DataFrame(ws_watch.get_all_records())
            user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
            target = st.selectbox("我的清單", user_stocks if user_stocks else ["2330.TW"])
            
            if user_stocks and st.button(f"🗑️ 刪除 {target}", use_container_width=True):
                rows = ws_watch.get_all_values()
                for i, row in enumerate(rows):
                    if i > 0 and row[0] == st.session_state.user and row[1] == target:
                        ws_watch.delete_rows(i + 1); st.rerun()
            
            st.divider()
            new_s = st.text_input("新增代碼").strip().upper()
            if st.button("➕ 確認新增", use_container_width=True):
                if new_s and new_s not in user_stocks:
                    ws_watch.append_row([st.session_state.user, new_s]); st.rerun()
            
            st.divider()
            unit = st.selectbox("時間單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測天數", 1, 30, 7)
            if st.button("🚪 登出", use_container_width=True):
                st.session_state.user = None; st.rerun()

        show_ultimate_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
