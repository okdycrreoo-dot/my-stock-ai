import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 高級技術分析平台", layout="wide")

# --- 2. 安全連線核心 (保持不變) ---
@st.cache_resource
def get_google_client():
    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"連線失敗: {e}")
        return None

# --- 3. 技術分析繪圖模組 ---
def show_advanced_analysis(symbol, period_choice, precision):
    try:
        period_map = {"5天": "1mo", "1個月": "3mo", "半年": "1y", "一年": "2y"}
        stock = yf.Ticker(symbol)
        df = stock.history(period=period_map[period_choice])
        
        if df.empty:
            st.error(f"找不到代碼 {symbol} 的數據")
            return

        # --- 技術指標計算 ---
        # 均線 (MA)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        # 布林通道 (BB)
        std = df['Close'].rolling(window=20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        # RSI (相對強弱指數)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- 創建複合圖表 (3個區域: K線/技術線, 交易量, RSI) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, 
                           row_heights=[0.6, 0.2, 0.2])

        # 1. K線圖 & 均線 & 布林通道
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='cyan', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_up'], name='布林上軌', line=dict(color='gray', dash='dash'), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='布林下軌', line=dict(color='gray', dash='dash'), opacity=0.5), row=1, col=1)

        # AI 預測模擬
        future_days = 5
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        trend = 0.01 * (precision / 100)
        future_prices = [df['Close'].iloc[-1] * (1 + (i * trend)) for i in range(1, future_days + 1)]
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='AI 預測線', line=dict(color='orange', width=3, dash='dot')), row=1, col=1)

        # 2. 交易量 (Volume)
        colors = ['red' if df['Open'].iloc[i] > df['Close'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='交易量', marker_color=colors), row=2, col=1)

        # 3. RSI 指標
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI(14)', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"分析失敗: {e}")

# --- 4. 自選股管理 (含重複/30筆上限/刪除功能) ---
def manage_watchlist(client, user):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 自選股管理 (上限30筆)")
    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    sh = client.open_by_url(url)
    try:
        ws = sh.worksheet("watchlist")
    except:
        ws = sh.add_worksheet(title="watchlist", rows="1000", cols="5")
        ws.append_row(["username", "stock_symbol"])
    
    all_data = pd.DataFrame(ws.get_all_records())
    user_list = all_data[all_data['username'] == user]['stock_symbol'].tolist() if not all_data.empty else []
    
    # 新增
    new_stock = st.sidebar.text_input("輸入代碼", placeholder="例: AAPL", key="new_s").strip().upper()
    if st.sidebar.button("➕ 加入"):
        if new_stock in user_list: st.sidebar.error("已存在")
        elif len(user_list) >= 30: st.sidebar.error("已達上限")
        elif new_stock:
            ws.append_row([user, new_stock])
            st.rerun()

    # 刪除
    if user_list:
        del_target = st.sidebar.selectbox("選取刪除目標", ["請選擇"] + user_list)
        if st.sidebar.button("🗑️ 執行刪除") and del_target != "請選擇":
            cells = ws.findall(user)
            for c in cells:
                if ws.row_values(c.row)[1] == del_target:
                    ws.delete_rows(c.row)
                    st.rerun()
                    break
    return user_list

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    client = get_google_client()
    if not client: return

    if st.session_state.user is None:
        st.title("🚀 StockAI 高級技術分析平台")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        with t1:
            with st.form("l"):
                u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
                if st.form_submit_button("登入"):
                    df = pd.DataFrame(client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).get_all_records())
                    if not df[(df['username'].astype(str)==u) & (df['password'].astype(str)==p)].empty:
                        st.session_state.user = u
                        st.rerun()
        with t2:
            st.info("首次使用？請在此設定帳密")
            with st.form("s"):
                nu, np, cp = st.text_input("帳號"), st.text_input("密碼", type="password"), st.text_input("確認密碼", type="password")
                if st.form_submit_button("註冊"):
                    if np == cp and nu:
                        client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).append_row([nu, np])
                        st.success("註冊成功")
    else:
        st.sidebar.title(f"👤 {st.session_state.user}")
        stocks = manage_watchlist(client, st.session_state.user)
        target = st.sidebar.selectbox("選擇股票", stocks if stocks else ["2330.TW"])
        period = st.sidebar.radio("區間", ["5天", "1個月", "半年", "一年"])
        precision = st.sidebar.slider("AI 靈敏度", 0, 100, 50)
        if st.sidebar.button("登出"):
            st.session_state.user = None
            st.rerun()

        st.title(f"📊 技術分析儀表板: {target}")
        show_advanced_analysis(target, period, precision)

if __name__ == "__main__":
    main()
