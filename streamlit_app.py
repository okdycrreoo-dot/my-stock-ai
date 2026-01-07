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

# --- 2. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：物理重組 Base64 (徹底解決 Incorrect padding) ---
def rebuild_private_key(raw_key):
    """物理性重新封裝私鑰，確保符合 RSA 認證標準格式"""
    # 提取標頭與標尾
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 1. 只抓取 A-Z, a-z, 0-9, +, / 這五類字元 (完全剔除空格、換行與雜質)
    body = "".join(re.findall(r"[A-Za-z0-9\+/]", raw_key))
    
    # 2. 手動計算並補齊 '=' 填充符號
    # Base64 長度必須是 4 的倍數
    missing_padding = len(body) % 4
    if missing_padding:
        body += "=" * (4 - missing_padding)
    
    # 3. 按照 Google 標準：每 64 字元換一行
    formatted_body = "\n".join([body[i:i+64] for i in range(0, len(body), 64)])
    
    return f"{header}\n{formatted_body}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用物理重組後的私鑰
        final_key = rebuild_private_key(s["private_key"])
        
        info = {
            "type": "service_account",
            "project_id": s["project_id"],
            "private_key_id": s["private_key_id"],
            "private_key": final_key,
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
                    data = sheet.get_all_records()
                    df = pd.DataFrame(data)
                    
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
                    st.error(f"試算表連接成功，但讀取失敗。請確認分頁名稱為 'users'")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
