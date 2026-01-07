import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import base64

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 共享資源載入 (TensorFlow) ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：手動 Base64 填充與認證 ---
@st.cache_resource
def get_gspread_client():
    try:
        # 從 Secrets 獲取原始設定
        conf = st.secrets["connections"]["gsheets"].to_dict()
        
        # 關鍵：修正私鑰格式
        raw_key = conf.get("private_key", "")
        # 1. 處理轉義換行 2. 移除前後所有空格或隱形字元
        fixed_key = raw_key.replace("\\n", "\n").strip()
        
        # 重新構建標準認證字典
        creds_info = {
            "type": "service_account",
            "project_id": conf.get("project_id"),
            "private_key_id": conf.get("private_key_id"),
            "private_key": fixed_key,
            "client_email": conf.get("client_email"),
            "client_id": conf.get("client_id"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": conf.get("client_x509_cert_url")
        }
        
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 使用底層庫直接認證，避開 st.connection 的自動檢查 Bug
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
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
                    # 獲取試算表網址並讀取
                    sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(sheet_url).worksheet("users")
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    # 清理與比對
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
    st.divider()
    # 後續選股功能...
