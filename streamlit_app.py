import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 共享資源載入 (TensorFlow) ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：手動 Base64 填充與認證 ---
def pad_base64(data):
    """手動校準 Base64 長度，解決 65 字元報錯問題"""
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    return data

@st.cache_resource
def get_gspread_client():
    try:
        # 1. 取得原始 Secrets
        s = st.secrets["connections"]["gsheets"]
        
        # 2. 強制清洗 Private Key
        raw_key = s["private_key"].replace("\\n", "\n").replace("\n", "\n").strip()
        
        # 3. 重新封裝認證字典
        info = {
            "type": "service_account",
            "project_id": s["project_id"],
            "private_key_id": s["private_key_id"],
            "private_key": raw_key,
            "client_email": s["client_email"],
            "client_id": s["client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": s["client_x509_cert_url"]
        }
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # 使用底層庫直接認證，避開 st.connection
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線失敗：憑證格式不正確。\n錯誤詳情：{str(e)}")
        return None

# --- 4. 登入邏輯 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
    with st.form("login_panel"):
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        if st.form_submit_button("進入系統", use_container_width=True):
            client = get_gspread_client()
            if client:
                try:
                    sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(sheet_url).worksheet("users")
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
                    st.error(f"資料庫讀取失敗：{str(e)}")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的個人面板")
    st.info(f"系統狀態：{model_status}")
