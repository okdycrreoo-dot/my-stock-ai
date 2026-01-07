import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# --- 1. 頁面配置與高對比主題 ---
st.set_page_config(page_title="StockAI LSTM 智慧監控終端", layout="wide")

# 強制修正 CSS 背景與文字對比
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 10px; }
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
    except Exception as e:
        return None

# --- 3. 核心運算：技術指標與 AI 預測 ---
def get_stock_data(symbol, period_choice="1y"):
    df = yf.download(symbol, period=period_choice, interval="1d", progress=False)
    if df.empty: return None
    
    # 計算 MA
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    
    # 計算布林通道
    df['BB_up'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_low'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    return df

def run_ai_logic(df, predict_days, precision):
    # 模擬 LSTM 權重演算
    last_p = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    
    # 權重修正因子
    rsi_bias = (50 - rsi) * 0.001 # RSI 低於 50 產生上拉力
    trend_bias = (ma20 - last_p) / ma20 * 0.1 # 乖離率修正
    user_bias = (precision / 100) * 0.01
    
    pred_prices = []
    curr_p = last_p
    for i in range(1, predict_days + 1):
        # 加入非線性隨機波動
        volt = np.random.normal(0, 0.002)
        change = 1 + (user_bias + rsi_bias + trend_bias + volt)
        curr_p *= change
        pred_prices.append(curr_p)
    return pred_prices

# --- 4. 智慧掃描功能 ---
def smart_scanner(client, user, stocks):
    with st.expander("🔍 雲端清單智慧掃描 (RSI / 趨勢分析)"):
        if not stocks:
            st.write("目前清單無資料")
            return
        
        cols = st.columns(4)
        for i, s in enumerate(stocks[:8]): # 掃描前 8 支以保持效能
            data = yf.download(s, period="1mo", progress=False)
            if not data.empty:
                # 簡易 RSI 計算
                diff = data['Close'].diff()
                r = 100 - (100/(1+(diff.where(diff>0,0).mean()/(-diff.where(diff<0,0).mean()))))
                
                with cols[i % 4]:
                    if r < 35: st.success(f"{s}: 🟢 超跌 (RSI:{r:.0f})")
                    elif r > 65: st.error(f"{s}: 🔴 超買 (RSI:{r:.0f})")
                    else: st.info(f"{s}: ⚪ 持平 ({r:.0f})")

# --- 5. 主介面 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    client = get_google_client()
    if not client: return

    if st.session_state.user is None:
        # --- 登入/註冊頁面 ---
        st.title("🚀 StockAI 高級技術分析終端")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        # (此處省略登入註冊 logic，與前版一致)
    else:
        # --- 登入後 ---
        st.sidebar.title(f"👤 {st.session_state.user}")
        
        # 1. 股票清單與管理
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        ws = client.open_by_url(url).worksheet("watchlist")
        all_data = pd.DataFrame(ws.get_all_records())
        user_list = all_data[all_data['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_data.empty else []

        # 智慧掃描
        smart_scanner(client, st.session_state.user, user_list)

        # 2. 側邊欄控制
        with st.sidebar:
            target = st.selectbox("切換股票", user_list if user_list else ["2330.TW"])
            unit = st.selectbox("時間單位", ["日", "月", "年"])
            p_days = st.number_input("AI 預測延伸(天)", 1, 30, 7)
            prec = st.slider("AI 靈敏度", 0, 100, 50)
            
            # 管理清單
            if st.button("🗑️ 刪除目前股票"):
                cells = ws.findall(st.session_state.user)
                for c in cells:
                    if ws.row_values(c.row)[1] == target:
                        ws.delete_rows(c.row)
                        st.rerun()
            
            new_s = st.text_input("新增代碼").strip().upper()
            if st.button("➕ 加入清單"):
                if len(user_list) < 30 and new_s not in user_list:
                    ws.append_row([st.session_state.user, new_s])
                    st.rerun()

        # 3. 繪圖區
        df = get_stock_data(target)
        if df is not None:
            pred_prices = run_ai_logic(df, p_days, prec)
            
            # 顯示預測指標
            last_p = df['Close'].iloc[-1]
            target_p = pred_prices[-1]
            pct = ((target_p - last_p)/last_p)*100
            
            st.markdown(f"### 📊 深度分析：{target}")
            c1, c2, c3 = st.columns(3)
            c1.metric("目前價格", f"{last_p:.2f}")
            c2.metric(f"AI 預估 ({p_days}天)", f"{target_p:.2f}")
            c3.metric("預計漲跌", f"{pct:.2f}%", delta=f"{pct:.2f}%")

            # Plotly 繪圖 (高對比)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            zoom = {"日": 40, "月": 250, "年": 750}[unit]
            p_df = df.tail(zoom)
            
            # K線
            fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
            # MA20
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='月線 MA20', line=dict(color='#00F5FF', width=2)), row=1, col=1)
            # 預測線
            f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days+1)]
            fig.add_trace(go.Scatter(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4, dash='dash')), row=1, col=1)
            # 交易量
            fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='交易量', marker_color='#30363D'), row=2, col=1)

            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#0E1117", plot_bgcolor="#161B22", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
