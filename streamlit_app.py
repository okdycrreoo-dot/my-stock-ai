import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import time
from datetime import datetime

# ⚠️ 從 cron_job.py 引入引擎 (請確保 cron_job.py 也在根目錄)
try:
    from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context
except ImportError:
    st.error("找不到 cron_job.py，請確保檔案已上傳至 GitHub 根目錄。")

# =================================================================
# 段落 1：頁面初始化與手機版優化 (無側邊欄)
# =================================================================
st.set_page_config(
    page_title="Oracle AI 股市終端",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed" 
)

# 強制隱藏側邊欄的 CSS (手機版更乾淨)
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .stButton button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# 段落 2：資料庫連線邏輯
# =================================================================
@st.cache_resource
def get_db():
    creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        st.error("請在 Streamlit Secrets 設定 GCP_SERVICE_ACCOUNT_JSON")
        return None
    
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open("users")
    return {
        "user_ws": sh.worksheet("users"),
        "watch_ws": sh.worksheet("watchlist"),
        "pred_ws": sh.worksheet("predictions")
    }

# =================================================================
# 段落 3：會員系統 (兼容現有 users 表格)
# =================================================================
def auth_section(db):
    st.title("🔮 Oracle AI 終端")
    tab1, tab2 = st.tabs(["登入系統", "新帳號註冊"])
    
    with tab1:
        u = st.text_input("帳號", key="login_u")
        p = st.text_input("密碼", type="password", key="login_p")
        if st.button("立即進入"):
            users = db["user_ws"].get_all_records()
            found = next((row for row in users if str(row['username']) == u and str(row['password']) == p), None)
            if found:
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.rerun()
            else:
                st.error("帳號或密碼不正確")

    with tab2:
        new_u = st.text_input("設定帳號", key="reg_u")
        new_p = st.text_input("設定密碼", type="password", key="reg_p")
        if st.button("確認註冊"):
            users = db["user_ws"].get_all_records()
            if any(str(row['username']) == new_u for row in users):
                st.warning("帳號已存在")
            elif new_u and new_p:
                db["user_ws"].append_row([new_u, new_p])
                st.success("註冊成功，請切換至登入分頁")
            else:
                st.error("請填寫完整資訊")

# =================================================================
# 段落 4：主程式功能 (手機直向排列)
# =================================================================
def main_app(db):
    # 頂部狀態列
    t1, t2 = st.columns([3, 1])
    t1.markdown(f"👤 **{st.session_state['user']}**")
    if t2.button("登出", key="logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.divider()

    # 1. 獲取該使用者的專屬清單 (對應 image_499249.png)
    all_watch = db["watch_ws"].get_all_records()
    my_stocks = [r['symbol'] for r in all_watch if str(r['username']) == st.session_state['user']]
    
    if not my_stocks:
        st.info("您的追蹤清單目前是空的，請先在試算表加入股票代號。")
        return

    # 2. 下拉選單 (大面積按鈕感)
    target = st.selectbox("🎯 選擇觀測個股", ["請選擇股票"] + my_stocks)

    if target != "請選擇股票":
        # 讀取預測數據
        all_preds = db["pred_ws"].get_all_records()
        df_p = pd.DataFrame(all_preds)
        stock_data = df_p[df_p['symbol'] == target].tail(1)

        if stock_data.empty:
            st.warning(f"分析庫中尚無 {target} 的數據")
            if st.button(f"🚀 啟動即時 AI 診斷"):
                with st.spinner("AI 解析中..."):
                    df_yf, f_id = fetch_comprehensive_data(target)
                    mkt_df = fetch_market_context()
                    if df_yf is not None:
                        p_next, path_str, insight, biases, s_data, e_data = god_mode_engine(df_yf, f_id, mkt_df)
                        data_date = df_yf.index[-1].strftime("%Y-%m-%d")
                        # 構建寫入格式
                        upload_row = [data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待更新"] + s_data + [0] + [path_str, insight] + biases + e_data
                        db["pred_ws"].append_row(upload_row)
                        st.success("診斷成功！")
                        time.sleep(1)
                        st.rerun()
        else:
            # 3. 數據展示 (針對手機寬度設計)
            row = stock_data.iloc[0]
            
            c1, c2 = st.columns(2)
            c1.metric("預測價", f"${row['pred_close']}")
            c2.metric("盈虧比", row['rr_ratio'])
            
            c3, c4 = st.columns(2)
            c3.metric("情緒", row['sentiment'])
            c4.metric("基準日", row['date'])

            st.success(f"🤖 **AI 診斷語句：**\n\n{row['ai_insight']}")
            
            # 趨勢圖 (自動適應寬度)
            st.write("📈 **未來趨勢模擬路徑**")
            path_vals = [float(x) for x in str(row['pred_path']).split(',')]
            st.line_chart(path_vals)

# =================================================================
# 段落 5：主入口
# =================================================================
if __name__ == "__main__":
    db_con = get_db()
    if db_con:
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        
        if not st.session_state["logged_in"]:
            auth_section(db_con)
        else:
            main_app(db_con)
