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

# --- 1. 頁面配置與深色高對比主題 ---
st.set_page_config(page_title="StockAI 專業分析終端", layout="wide")

# 強制修正背景色與圖表區隔
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363D; }
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
        st.error(f"連線失敗: {e}")
        return None

# --- 3. 高級技術分析與自定義預測 ---
def show_advanced_analysis(symbol, unit_choice, predict_days, precision):
    try:
        # 根據單位選擇獲取歷史數據
        unit_map = {"日": "1d", "月": "1mo", "年": "1y"}
        # 為了計算指標，我們抓取足夠長度的數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="max" if unit_choice == "年" else "2y", interval="1d")
        
        if df.empty:
            st.error(f"找不到代碼 {symbol} 的數據")
            return

        # 根據使用者選擇縮放 X 軸顯示範圍
        zoom_map = {"日": 30, "月": 365, "年": 1095} # 顯示最近 X 天的數據
        plot_df = df.tail(zoom_map[unit_choice])

        # 計算技術指標
        plot_df['MA5'] = plot_df['Close'].rolling(window=5).mean()
        plot_df['MA20'] = plot_df['Close'].rolling(window=20).mean()
        std = plot_df['Close'].rolling(window=20).std()
        plot_df['BB_up'] = plot_df['MA20'] + (std * 2)
        plot_df['BB_low'] = plot_df['MA20'] - (std * 2)
        
        # --- 預測邏輯 ---
        last_price = plot_df['Close'].iloc[-1]
        last_date = plot_df.index[-1]
        # 使用者輸入的預測長度（以日為單位延伸）
        future_dates = [last_date + timedelta(days=i) for i in range(1, predict_days + 1)]
        trend = 0.005 * (precision / 100) # 靈敏度影響斜率
        pred_prices = [last_price * (1 + (i * trend)) for i in range(1, predict_days + 1)]
        target_price = pred_prices[-1]
        total_change = ((target_price - last_price) / last_price) * 100

        # --- 頂部預測數據卡片 ---
        st.markdown(f"### 🎯 AI 預測分析 ({predict_days}天後)")
        c1, c2, c3 = st.columns(3)
        c1.metric("當前收盤價", f"{last_price:.2f}")
        c2.metric("預估目標價", f"{target_price:.2f}")
        c3.metric("預計總漲跌幅", f"{total_change:.2f}%", f"{total_change:.2f}%")

        # --- 創建複合圖表 (高對比度設定) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.08, row_heights=[0.7, 0.3])

        # 1. K線與技術線 (背景加深以凸顯線條)
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
        
        # 使用高飽和度顏色區分
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA5'], name='MA5', line=dict(color='#FFD700', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_up'], name='布林上軌', line=dict(color='#808080', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_low'], name='布林下軌', line=dict(color='#808080', dash='dot')), row=1, col=1)

        # 2. 延伸 AI 預測線 (明顯亮橘色)
        fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name='AI 預測路徑', 
                                 line=dict(color='#FF4500', width=3, dash='dashdot')), row=1, col=1)

        # 3. 交易量
        vol_colors = ['#FF3131' if plot_df['Open'].iloc[i] > plot_df['Close'].iloc[i] else '#00FF41' for i in range(len(plot_df))]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='交易量', marker_color=vol_colors), row=2, col=1)

        # 圖表樣式修正：增加對比度與網格線可見度
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117", # 與網頁背景一致
            plot_bgcolor="#161B22",  # 圖表內部稍淺，增加層次感
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"繪圖失敗: {e}")

# --- 4. 自選股管理 (含30筆上限與刪除) ---
def manage_watchlist(client, user):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 自選清單管理")
    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    sh = client.open_by_url(url)
    ws = sh.worksheet("watchlist")
    
    all_data = pd.DataFrame(ws.get_all_records())
    user_list = all_data[all_data['username'] == user]['stock_symbol'].tolist() if not all_data.empty else []
    
    # 新增
    with st.sidebar.expander("➕ 新增股票"):
        new_s = st.text_input("輸入代碼", placeholder="AAPL", key="ns").strip().upper()
        if st.button("確認加入"):
            if new_s in user_list: st.warning("已在清單中")
            elif len(user_list) >= 30: st.error("已達30筆上限")
            elif new_s:
                ws.append_row([user, new_s])
                st.rerun()

    # 刪除
    if user_list:
        with st.sidebar.expander("🗑️ 刪除股票"):
            ds = st.selectbox("選擇要刪除的股票", user_list)
            if st.button("確認執行刪除"):
                cells = ws.findall(user)
                for c in cells:
                    if ws.row_values(c.row)[1] == ds:
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
        st.title("🚀 StockAI 高級技術分析終端")
        t1, t2 = st.tabs(["🔑 登入系統", "📝 快速註冊"])
        with t1:
            with st.form("l"):
                u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
                if st.form_submit_button("登入"):
                    df = pd.DataFrame(client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).get_all_records())
                    if not df[(df['username'].astype(str)==u) & (df['password'].astype(str)==p)].empty:
                        st.session_state.user = u
                        st.rerun()
        with t2:
            with st.form("s"):
                nu, np, cp = st.text_input("註冊帳號"), st.text_input("密碼", type="password"), st.text_input("確認密碼", type="password")
                if st.form_submit_button("註冊"):
                    if np == cp and nu:
                        client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).get_worksheet(0).append_row([nu, np])
                        st.success("註冊成功，請登入")
    else:
        # --- 登入後控制面板 ---
        st.sidebar.title(f"👤 {st.session_state.user}")
        
        # 1. 時間單位選擇 (日、月、年)
        unit = st.sidebar.selectbox("圖表 X 軸單位", ["日", "月", "年"])
        
        # 2. 自定義 AI 預測天數
        p_days = st.sidebar.number_input("AI 預測延伸天數", min_value=1, max_value=30, value=7)
        
        # 3. 靈敏度
        prec = st.sidebar.slider("AI 預測靈敏度 (%)", 0, 100, 50)
        
        stocks = manage_watchlist(client, st.session_state.user)
        target = st.sidebar.selectbox("當前查看股票", stocks if stocks else ["2330.TW"])
        
        if st.sidebar.button("登出系統"):
            st.session_state.user = None
            st.rerun()

        # 主顯示區
        st.title(f"📊 技術分析：{target}")
        show_advanced_analysis(target, unit, p_days, prec)

if __name__ == "__main__":
    main()
