import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import time

# 安全連線建立
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

def login():
    st.title("🚀 StockAI 登入系統")
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        with st.form("login_form"):
            u = st.text_input("帳號")
            p = st.text_input("密碼", type="password")
            if st.form_submit_button("進入系統", use_container_width=True):
                client = get_google_client()
                if client:
                    try:
                        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                        # 自動抓取「第一個分頁」，避免分頁名稱錯誤
                        sh = client.open_by_url(url)
                        sheet = sh.get_worksheet(0) 
                        df = pd.DataFrame(sheet.get_all_records())
                        
                        # 驗證 (將欄位轉為字串並去除空白)
                        df.columns = df.columns.str.strip()
                        df['username'] = df['username'].astype(str).str.strip()
                        df['password'] = df['password'].astype(str).str.strip()
                        
                        check = df[(df['username'] == u) & (df['password'] == p)]
                        if not check.empty:
                            st.session_state.user = u
                            st.success("驗證通過！")
                            st.rerun()
                        else:
                            st.error("帳號或密碼錯誤")
                    except Exception as e:
                        st.error(f"資料讀取失敗，原因：{e}")
    else:
        st.write(f"歡迎回來，{st.session_state.user}")
        if st.button("登出"):
            st.session_state.user = None
            st.rerun()

login()
