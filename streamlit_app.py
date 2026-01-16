import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import time
from datetime import datetime
import hashlib

# ⚠️ 重要：從您的 cron_job.py 引入核心引擎
from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context

# =================================================================
# 段落 1：頁面配置與初始化
# =================================================================
st.set_page_config(
    page_title="Oracle AI 股市終端",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 密碼檢查邏輯：兼容明文與加密 (建議未來全面加密)
def check_password(input_pw, stored_pw):
    return str(input_pw) == str(stored_pw)

# =================================================================
# 段落 2：資料庫連線
# =================================================================
@st.cache_resource
def init_gspread():
    creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        st.error("❌ Secrets 中找不到 GCP_SERVICE_ACCOUNT_JSON")
        return None
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    return gspread.authorize(creds)

def get_db():
    client = init_gspread()
    sh = client.open("users")
    return {
        "user_table": sh.worksheet("users"),     # 對應 image_4991ce.png
        "watch_table": sh.worksheet("watchlist"), # 對應 image_499249.png
        "pred_table": sh.worksheet("predictions")
    }

# =================================================================
# 段落 3：登入與註冊系統 (手機優化版)
# =================================================================
def auth_section():
    db = get_db()
    st.title("🔮 Oracle AI 登入")
    
    tab1, tab2 = st.tabs(["帳號登入", "新用戶註冊"])
    
    with tab1:
        login_user = st.text_input("帳號", key="l_user")
        login_pw = st.text_input("密碼", type="password", key="l_pw")
        if st.button("立即登入", use_container_width=True):
            user_data = db["user_table"].get_all_records()
            # 比對現有表格資料
            found = next((u for u in user_data if str(u['username']) == login_user and check_password(login_pw, u['password'])), None)
            if found:
                st.session_state["logged_in"] = True
                st.session_state["user"] = login_user
                st.success("歡迎回來！")
                st.rerun()
            else:
                st.error("帳號或密碼錯誤，請重新輸入。")

    with tab2:
        st.subheader("建立新帳號")
        new_user = st.text_input("帳號名稱", key="n_user")
        new_pw = st.text_input("密碼設定", type="password", key="n_pw")
        if st.button("提交註冊", use_container_width=True):
            user_data = db["user_table"].get_all_records()
            if any(str(u['username']) == new_user for u in user_data):
                st.warning("⚠️ 此帳號已存在，請嘗試登入或更換名稱。")
            elif new_user and new_pw:
                db["user_table"].append_row([new_user, new_pw]) # 註冊存入 users 分頁
                st.success("註冊成功！請切換到登入分頁。")
            else:
                st.error("欄位不可為空。")

# =================================================================
# 段落 4：主程式面 (手機直立優化，無側邊欄)
# =================================================================
def main_app():
    db = get_db()
    current_user = st.session_state['user']
    
    # 頂部控制列
    col_user, col_logout = st.columns([3, 1])
    col_user.write(f"👤 **{current_user}**")
    if col_logout.button("登出", size="small"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.divider()

    # 1. 讀取並篩選該使用者的 Watchlist
    watch_data = db["watch_table"].get_all_records()
    user_symbols = [row['symbol'] for row in watch_data if str(row['username']) == current_user]
    
    if not user_symbols:
        st.warning("您目前還沒有追蹤任何股票。請先到試算表加入數據。")
        return

    # 2. 個股選擇器 (手機友善大選單)
    target = st.selectbox("🎯 選擇觀測個股", ["請選擇"] + user_symbols)

    if target != "請選擇":
        # 3. 讀取預測數據
        all_preds = db["pred_table"].get_all_records()
        df_p = pd.DataFrame(all_preds)
        
        # 找到該股票最新的一筆預測
        stock_pred = df_p[df_p['symbol'] == target].tail(1)

        if stock_pred.empty:
            st.warning(f"分析庫中目前沒有 {target} 的數據")
            if st.button(f"🚀 啟動即時 AI 診斷", use_container_width=True):
                with st.spinner("AI 正在解析市場大數據..."):
                    df_yf, f_id = fetch_comprehensive_data(target)
                    mkt_df = fetch_market_context()
                    if df_yf is not None:
                        p_next, path_str, insight, biases, s_data, e_data = god_mode_engine(df_yf, f_id, mkt_df)
                        data_date = df_yf.index[-1].strftime("%Y-%m-%d")
                        upload_row = [data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待更新"] + s_data + [0] + [path_str, insight] + biases + e_data
                        db["pred_table"].append_row(upload_row)
                        st.success("診斷完成！數據已入庫。")
                        time.sleep(1)
                        st.rerun()
        else:
            # 4. 數據展示面板
            row = stock_pred.iloc[0]
            
            # 關鍵數據卡 (手機雙排顯示)
            c1, c2 = st.columns(2)
            c1.metric("AI 預測價", f"${row['pred_close']}")
            c2.metric("盈虧比 (RR)", row['rr_ratio'])
            
            c3, c4 = st.columns(2)
            c3.metric("更新日期", row['date'])
            c4.metric("市場情緒", row['sentiment'])

            st.divider()
            
            # AI 診斷文本
            st.markdown("### 🤖 AI Oracle 綜合診斷")
            st.success(row['ai_insight'])
            
            # 簡易路徑預覽 (下階段我們改 Plotly 漂亮圖表)
            st.markdown("### 📈 預測趨勢路徑")
            path_values = [float(x) for x in str(row['pred_path']).split(',')]
            st.line_chart(path_values)

# =================================================================
# 段落 5：主入口
# =================================================================
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if not st.session_state["logged_in"]:
        auth_section()
    else:
        main_app()
