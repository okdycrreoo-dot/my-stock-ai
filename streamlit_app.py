import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import time
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 全功能管理平台", layout="wide")

# --- 2. 安全連線核心 ---
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

# --- 3. 核心功能模組 ---

# 功能 A: 即時行情與 AI 繪圖
def show_stock_analysis(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty:
            st.error("找不到該代碼數據")
            return

        # 計算基本指標
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = last_price - prev_price
        pct = (change / prev_price) * 100

        # 頂部數據卡片
        c1, c2, c3 = st.columns(3)
        c1.metric("當前股價", f"{last_price:.2f}")
        c2.metric("今日漲跌", f"{change:.2f}", f"{pct:.2f}%")
        c3.info(f"AI 模型狀態：運作中")

        # 繪製 Plotly 圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史價格', line=dict(color='#00ffcc')))
        # 模擬 AI 預測線 (未來 5 天)
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 6)]
        future_prices = [last_price * (1 + (i * 0.01)) for i in range(1, 6)] # 這裡可替換為您的 model.predict
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='AI 預測趨勢', line=dict(dash='dot', color='orange')))
        
        fig.update_layout(template="plotly_dark", title=f"{symbol} 趨勢分析圖", height=500)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"分析出錯: {e}")

# 功能 B: 雲端自選股管理 (與 Google Sheets 同步)
def manage_watchlist(client, user):
    st.subheader("📋 我的雲端自選股")
    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    
    # 假設您在試算表中有一個分頁叫 'watchlist'
    try:
        sh = client.open_by_url(url)
        try:
            ws = sh.worksheet("watchlist")
        except:
            ws = sh.add_worksheet(title="watchlist", rows="100", cols="20")
            ws.append_row(["username", "stock_symbol"])

        # 讀取當前使用者的股票
        all_data = pd.DataFrame(ws.get_all_records())
        user_list = []
        if not all_data.empty:
            user_list = all_data[all_data['username'] == user]['stock_symbol'].tolist()

        # 介面：新增與顯示
        col1, col2 = st.columns([3, 1])
        new_stock = col1.text_input("輸入要新增的代碼 (例: 2330.TW)", placeholder="2330.TW")
        if col2.button("新增至雲端") and new_stock:
            ws.append_row([user, new_stock])
            st.success("已同步至 Google Sheets")
            time.sleep(1)
            st.rerun()

        st.write("目前追蹤：", ", ".join(user_list) if user_list else "尚無資料")
        return user_list
    except Exception as e:
        st.error(f"自選股同步失敗: {e}")
        return []

# --- 4. 主程式邏輯 ---
def main():
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        # 登入頁面
        st.title("🚀 StockAI 登入系統")
        with st.form("login"):
            u = st.text_input("帳號")
            p = st.text_input("密碼", type="password")
            if st.form_submit_button("進入系統", use_container_width=True):
                client = get_google_client()
                if client:
                    sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
                    user_df = pd.DataFrame(sh.get_worksheet(0).get_all_records())
                    if not user_df[(user_df['username'].astype(str) == u) & (user_df['password'].astype(str) == p)].empty:
                        st.session_state.user = u
                        st.rerun()
                    else:
                        st.error("登入失敗")
    else:
        # 登入後的專業儀表板
        client = get_google_client()
        st.sidebar.title(f"👤 {st.session_state.user}")
        
        # 整合自選股到側邊欄
        stocks = manage_watchlist(client, st.session_state.user)
        target = st.sidebar.selectbox("切換觀看股票", stocks if stocks else ["2330.TW"])
        
        if st.sidebar.button("登出系統"):
            st.session_state.user = None
            st.rerun()

        # 主畫面顯示
        st.title(f"📊 股票分析終端: {target}")
        show_stock_analysis(target)

if __name__ == "__main__":
    main()
