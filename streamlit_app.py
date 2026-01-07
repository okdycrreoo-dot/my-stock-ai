import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re
import base64

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 核心修復：極簡且嚴格的私鑰修復 ---
def get_pure_private_key(raw_key):
    # 物理性移除所有非 Base64 合法字元
    core = "".join(re.findall(r"[A-Za-z0-9\+/]", raw_key))
    
    # 補足 Padding
    missing_padding = len(core) % 4
    if missing_padding:
        core += "=" * (4 - missing_padding)
    
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 每 64 個字元換一行
    formatted_body = "\n".join([core[i:i+64] for i in range(0, len(core), 64)])
    
    return f"{header}\n{formatted_body}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        # 讀取並轉為字典
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 進行最終格式化
        if "private_key" in s_dict:
            s_dict["private_key"] = get_pure_private_key(s_dict["private_key"])
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        # 顯示更詳細的報錯，幫助抓出是哪個欄位出問題
        st.error(f"安全性連線最終嘗試中: {str(e)}")
        if "private_key" in st.secrets["connections"]["gsheets"]:
            key_len = len(st.secrets["connections"]["gsheets"]["private_key"])
            st.warning(f"診斷訊息：偵測到私鑰長度為 {key_len} 字元。")
        return None

# --- 3. 登入邏輯 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
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
                    
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證通過！")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")
                except Exception as e:
                    st.error(f"資料讀取失敗：{str(e)}")

# --- 4. 主程式頁面 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.info("雲端連線狀態：正常 ✅")
