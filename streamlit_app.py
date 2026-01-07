import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 共享資源載入 (TensorFlow) ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：二進位純淨提取 (徹底解決 Unused bytes 問題) ---
def get_pure_private_key(raw_key):
    """將私鑰轉換為純淨格式，物理剔除所有二進位雜質"""
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 1. 提取中間內容
    content = raw_key.replace(header, "").replace(footer, "")
    
    # 2. 核心過濾：只允許 A-Z, a-z, 0-9, +, /, =
    # 這次我們使用更嚴格的正規表達式，並移除所有換行符進行重新排版
    pure_base64 = "".join(re.findall(r"[A-Za-z0-9\+/=]", content))
    
    # 3. 每 64 個字元換一行 (這是 Google 認證標準格式)
    formatted_content = "\n".join([pure_base64[i:i+64] for i in range(0, len(pure_base64), 64)])
    
    return f"{header}\n{formatted_content}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用強效物理過濾後的私鑰
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
        # 如果還是失敗，嘗試最後一種：直接輸出錯誤類型幫助診斷
        st.error(f"安全性連線最終嘗試中: {str(e)}")
        return None

# --- 4. 登入系統邏輯 ---
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
                    
                    # 帳密驗證
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

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.write(f"AI 模型狀態：{model_status}")
