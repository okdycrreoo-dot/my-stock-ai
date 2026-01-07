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
st.set_page_config(page_title="StockAI 管理平台", layout="wide")

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

def show_stock_analysis(symbol, period_choice, precision):
    try:
        period_map = {"5天": "1mo", "1個月": "3mo", "半年": "1y", "一年": "2y"}
        stock = yf.Ticker(symbol)
        df = stock.history(period=period_map[period_choice])
        if df.empty:
            st.error("找不到該代碼數據")
            return

        last_price = df['Close'].iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史價格', line=dict(color='#00ffcc')))
        
        # AI 預測模擬
        future_days = 10
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
        trend_factor = 0.01 * (precision / 100) 
        future_prices = [last_price * (1 + (i * trend_factor)) for i in range(1, future_days + 1)]
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name=f'AI 預測 (精度:{precision}%)', line=dict(dash='dot', color='orange')))
        fig.update_layout(template="plotly_dark", title=f"{symbol} 趨勢圖", height=500)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"分析失敗: {e}")

def manage_watchlist(client, user):
    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    sh = client.open_by_url(url)
    try:
        ws = sh.worksheet("watchlist")
    except:
        ws = sh.add_worksheet(title="watchlist", rows="100", cols="20")
        ws.append_row(["username", "stock_symbol"])
    
    all_data = pd.DataFrame(ws.get_all_records())
    user_list = all_data[all_data['username'] == user]['stock_symbol'].tolist() if not all_data.empty else []
    
    new_stock = st.sidebar.text_input("新增自選代碼", placeholder="例如: 2330.TW")
    if st.sidebar.button("加入清單"):
        ws.append_row([user, new_stock])
        st.success("同步成功")
        time.sleep(1)
        st.rerun()
    return user_list

# --- 4. 主程式邏輯 ---
def main():
    if 'user' not in st.session_state:
        st.session_state.user = None

    client = get_google_client()
    if not client: return

    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    sh = client.open_by_url(url)
    user_ws = sh.get_worksheet(0)

    if st.session_state.user is None:
        st.title("🚀 StockAI 全方位預測平台")
        
        # 使用 Tabs 分開登入與註冊
        tab_login, tab_signup = st.tabs(["🔑 帳號登入", "📝 帳號註冊"])

        with tab_login:
            with st.form("login_form"):
                u = st.text_input("帳號").strip()
                p = st.text_input("密碼", type="password").strip()
                if st.form_submit_button("立即登入", use_container_width=True):
                    user_df = pd.DataFrame(user_ws.get_all_records())
                    user_match = user_df[(user_df['username'].astype(str) == u) & (user_df['password'].astype(str) == p)]
                    if not user_match.empty:
                        st.session_state.user = u
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")

        with tab_signup:
            st.info("💡 首次使用？請在此設定您的專屬帳密")
            with st.form("signup_form"):
                new_u = st.text_input("設定新帳號").strip()
                new_p = st.text_input("設定新密碼", type="password").strip()
                confirm_p = st.text_input("確認新密碼", type="password").strip()
                if st.form_submit_button("完成註冊並登入", use_container_width=True):
                    user_df = pd.DataFrame(user_ws.get_all_records())
                    if new_u in user_df['username'].astype(str).values:
                        st.error("此帳號已被註冊，請換一個名字")
                    elif new_p != confirm_p:
                        st.error("兩次密碼輸入不一致")
                    elif new_u and new_p:
                        user_ws.append_row([new_u, new_p])
                        st.success("註冊成功！")
                        st.session_state.user = new_u
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("帳號密碼不可為空")
    else:
        # --- 登入後的面板 ---
        st.sidebar.title(f"👤 {st.session_state.user}")
        period = st.sidebar.radio("歷史區間", ["5天", "1個月", "半年", "一年"])
        precision = st.sidebar.slider("AI 預測靈敏度", 0, 100, 50)
        
        stocks = manage_watchlist(client, st.session_state.user)
        target = st.sidebar.selectbox("切換觀看股票", stocks if stocks else ["2330.TW"])
        
        if st.sidebar.button("登出"):
            st.session_state.user = None
            st.rerun()

        st.title(f"📊 分析面板: {target}")
        show_stock_analysis(target, period, precision)

if __name__ == "__main__":
    main()
