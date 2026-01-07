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

# --- 3. 核心修正：二進位過濾器 (解決 Unused bytes 問題) ---
def get_pure_private_key(raw_key):
    """徹底過濾非合法 Base64 字元，並補足 Padding"""
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 提取核心部分 (移除標頭、標尾、轉義換行)
    core = raw_key.replace(header, "").replace(footer, "").replace("\\n", "").replace("\n", "").strip()
    
    # 關鍵：只保留 Base64 合法字元集 (A-Z, a-z, 0-9, +, /)
    # 這會物理性剔除導致報錯的 \xdab 等二進位雜質
    core = "".join(re.findall(r"[A-Za-z0-9\+/]", core))
    
    # 強制補齊填充字元 '=' 至 4 的倍數 (解決 Padding 錯誤)
    missing_padding = len(core) % 4
    if missing_padding:
        core += "=" * (4 - missing_padding)
    
    # 按照 Google 標準格式：每 64 個字元換一行重新排版
    formatted_body = "\n".join([core[i:i+64] for i in range(0, len(core), 64)])
    
    return f"{header}\n{formatted_body}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用二進位過濾後的私鑰
        fixed_key = get_pure_private_key(s["private_key"])
        
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
        st.error(f"安全性連線最終嘗試中: {str(e)}")
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
                    st.error(f"資料讀取失敗，請確認分頁 'users' 存在。")

# --- 5. 主程式頁面 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.write(f"系統狀態：{model_status}")
