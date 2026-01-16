import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import time
import sys
from datetime import datetime

# =================================================================
# 段落 1：頁面配置與路徑修復 (確保能讀到 cron_job.py)
# =================================================================
st.set_page_config(
    page_title="Oracle AI 股市終端",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed" 
)

# 手機版 UI 優化：隱藏側邊欄，按鈕滿版
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .stButton button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 確保程式能找到同目錄下的 cron_job.py
sys.path.append(os.path.dirname(__file__))

try:
    from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context
except ImportError as e:
    st.error(f"⚠️ 引擎加載失敗，請檢查 cron_job.py 是否在 GitHub 根目錄。錯誤: {e}")

# =================================================================
# 段落 2：資料庫連線 (使用現代化 google-auth)
# =================================================================
@st.cache_resource
def get_db():
    # 從 Streamlit Secrets 讀取憑證
    creds_info = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_info:
        st.error("❌ 請在 Streamlit Secrets 設定 GCP_SERVICE_ACCOUNT_JSON")
        return None
    
    try:
        info = json.loads(creds_info)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(info, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open("users")
        return {
            "user_ws": sh.worksheet("users"),
            "watch_ws": sh.worksheet("watchlist"),
            "pred_ws": sh.worksheet("predictions")
        }
    except Exception as e:
        st.error(f"連線 Google Sheets 失敗: {e}")
        return None

# =================================================================
# 段落 3：會員系統 (兼容您現有的 users 表格)
# =================================================================
def auth_section(db):
    st.title("🔮 Oracle AI 終端")
    tab1, tab2 = st.tabs(["登入系統", "註冊帳號"])
    
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
                st.error("帳號或密碼錯誤")

    with tab2:
        new_u = st.text_input("設定帳號", key="reg_u")
        new_p = st.text_input("設定密碼", type="password", key="reg_p")
        if st.button("確認註冊"):
            users = db["user_ws"].get_all_records()
            if any(str(row['username']) == new_u for row in users):
                st.warning("此帳號已被使用")
            elif new_u and new_p:
                db["user_ws"].append_row([new_u, new_p])
                st.success("註冊成功！請切換至登入分頁。")
            else:
                st.error("欄位不可為空")

# =================================================================
# 段落 4：主功能介面 (手機直向優化)
# =================================================================
def main_app(db):
    t1, t2 = st.columns([3, 1])
    t1.markdown(f"👤 **{st.session_state['user']}**")
    if t2.button("登出"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.divider()

    # 1. 讀取專屬清單
    all_watch = db["watch_ws"].get_all_records()
    my_stocks = [r['symbol'] for r in all_watch if str(r['username']) == st.session_state['user']]
    
    if not my_stocks:
        st.info("您的清單目前為空。")
        return

    # 2. 選股與預測
    target = st.selectbox("🎯 選擇觀測個股", ["請選擇"] + my_stocks)

    if target != "請選擇":
        all_preds = db["pred_ws"].get_all_records()
        df_p = pd.DataFrame(all_preds)
        stock_data = df_p[df_p['symbol'] == target].tail(1)

        if stock_data.empty:
            st.warning(f"目前尚無 {target} 的數據")
            if st.button(f"🚀 啟動即時 AI 診斷"):
                with st.spinner("AI 正在解析數據..."):
                    df_yf, f_id = fetch_comprehensive_data(target)
                    mkt_df = fetch_market_context()
                    if df_yf is not None:
                        # 呼叫 cron_job.py 引擎
                        p_next, path_str, insight, biases, s_data, e_data = god_mode_engine(df_yf, f_id, mkt_df)
                        data_date = df_yf.index[-1].strftime("%Y-%m-%d")
                        upload_row = [data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待更新"] + s_data + [0] + [path_str, insight] + biases + e_data
                        db["pred_ws"].append_row(upload_row)
                        st.success("診斷完成！")
                        st.rerun()
        else:
            # 展示數據
            row = stock_data.iloc[0]
            m1, m2 = st.columns(2)
            m1.metric("預測價", f"${row['pred_close']}")
            m2.metric("盈虧比", row['rr_ratio'])
            
            st.success(f"🤖 **AI 診斷：**\n\n{row['ai_insight']}")
            
            # 簡易圖表
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
