import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import time

st.set_page_config(page_title="StockAI 管理系統", layout="centered")

@st.cache_resource
def get_stable_client():
    try:
        # 1. 取得 Secrets 裡的 service_account 字串並解碼為字典
        service_account_str = st.secrets["connections"]["gsheets"]["service_account"]
        creds_info = json.loads(service_account_str)
        
        # 2. 定義權限範圍
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 3. 建立認證與客戶端
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"連線失敗：{e}")
        return None

# --- 登入邏輯 ---
def login():
    st.title("🚀 StockAI 登入系統")
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        with st.form("login_form"):
            u = st.text_input("帳號")
            p = st.text_input("密碼", type="password")
            if st.form_submit_button("進入系統", use_container_width=True):
                client = get_stable_client()
                if client:
                    try:
                        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                        sheet = client.open_by_url(url).worksheet("users")
                        df = pd.DataFrame(sheet.get_all_records())
                        
                        # 驗證帳密
                        check = df[(df['username'].astype(str) == u) & (df['password'].astype(str) == p)]
                        if not check.empty:
                            st.session_state.user = u
                            st.rerun()
                        else:
                            st.error("帳號或密碼不正確")
                    except Exception as e:
                        st.error(f"資料存取失敗: {e}")
    else:
        st.success(f"已登入：{st.session_state.user}")
        if st.button("登出"):
            st.session_state.user = None
            st.rerun()

login()
