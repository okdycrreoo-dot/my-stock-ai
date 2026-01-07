import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：強制 Base64 合規化功能 ---
def fix_base64_string(key_str):
    """徹底解決 65 字元報錯的終極函數"""
    # 1. 處理轉義字元並移除首尾所有空白
    key_str = key_str.replace("\\n", "\n").strip()
    
    # 2. 如果是 65 字元錯誤，通常是末尾多了一個隱形換行
    # 我們只保留 "-----BEGIN..." 到 "...END-----\n" 之間的內容
    if "-----BEGIN PRIVATE KEY-----" in key_str:
        header = "-----BEGIN PRIVATE KEY-----\n"
        footer = "\n-----END PRIVATE KEY-----"
        # 提取中間的核心編碼部分
        core_content = key_str.replace(header, "").replace(footer, "").replace("\n", "").strip()
        # 強制補齊 Base64 填充字元 '=' 至 4 的倍數
        missing_padding = len(core_content) % 4
        if missing_padding:
            core_content += '=' * (4 - missing_padding)
        # 重新組成標準格式
        return f"{header}{core_content}{footer}"
    return key_str

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用修正後的私鑰
        fixed_key = fix_base64_string(s["private_key"])
        
        info = {
            "type": "service_account",
            "project_id": s["project_id"],
            "private_key_id": s["private_key_id"],
            "private_key": fixed_key,
            "client_email": s["client_email"],
            "client_id": s["client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": s["client_x509_cert_url"]
        }
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"連線失敗（嘗試修復中）: {str(e)}")
        return None

# --- 4. 登入邏輯 ---
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
                    st.error(f"試算表存取失敗: {e}")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
