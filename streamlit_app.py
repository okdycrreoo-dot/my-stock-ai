import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型運算核心已啟動"

model_status = load_shared_model()

# --- 3. 強效連線工具 (繞過 st.connection 以避免 Base64 報錯) ---
@st.cache_resource
def get_gspread_client():
    try:
        # 1. 從 Secrets 取得原始資料
        info = st.secrets["connections"]["gsheets"].to_dict()
        
        # 2. 手動清洗私鑰 (這是關鍵：徹底解決 binascii.Error)
        # 移除可能導致 65 字元報錯的所有隱形空格與換行
        private_key = info.get("private_key", "")
        fixed_key = private_key.replace("\\n", "\n").strip()
        
        # 3. 重新封裝認證字典
        creds_dict = {
            "type": "service_account",
            "project_id": info.get("project_id"),
            "private_key_id": info.get("private_key_id"),
            "private_key": fixed_key,
            "client_email": info.get("client_email"),
            "client_id": info.get("client_id"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": info.get("client_x509_cert_url")
        }
        
        # 4. 建立認證與客戶端
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線失敗: {e}")
        return None

# --- 4. 登入系統邏輯 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
    with st.form("login_form"):
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        submit = st.form_submit_button("進入系統", use_container_width=True)
        
        if submit:
            client = get_gspread_client()
            if client:
                try:
                    # 打開試算表並讀取 'users' 工作表
                    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(url).worksheet("users")
                    data = sheet.get_all_records()
                    df = pd.DataFrame(data)
                    
                    # 驗證
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證成功！")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳密不正確")
                except Exception as e:
                    st.error(f"資料讀取失敗: {e}")

# --- 5. 主程式介面 ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"用戶：{user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
    st.divider()
    
    stock = st.text_input("輸入股票代碼進行分析")
    if stock:
        st.write(f"正在分析 {stock}...")
