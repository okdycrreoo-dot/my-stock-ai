import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 共享資源 (TensorFlow) ---
@st.cache_resource
def load_shared_model():
    # 確保 30 人併發時模型只載入一次
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：手動 Base64 補齊與認證 ---
@st.cache_resource
def get_gspread_client():
    try:
        # 1. 取得原始 Secrets
        s = st.secrets["connections"]["gsheets"]
        
        # 2. 清洗 Private Key：處理轉義換行並移除所有首尾不可見字元
        # 這是解決截圖中 "Invalid base64-encoded string (65)" 的關鍵
        raw_key = s["private_key"].replace("\\n", "\n").strip()
        
        # 3. 構建認證字典 (不使用 st.connection 避免自動檢查報錯)
        creds_dict = {
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
        
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 使用底層庫直接認證
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線失敗：憑證格式不正確。\n詳細訊息：{str(e)}")
        return None

# --- 4. 登入系統 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
    with st.form("login_gate"):
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        if st.form_submit_button("進入系統", use_container_width=True):
            client = get_gspread_client()
            if client:
                try:
                    # 讀取試算表
                    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(url).worksheet("users")
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    # 數據清洗與比對
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
                    st.error(f"無法存取試算表，請確認分頁名稱為 'users'。錯誤：{e}")

# --- 5. 主程式頁面 ---
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
    # 這裡放選股分析功能
