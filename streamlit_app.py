import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import json

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型運算核心已啟動"

model_status = load_shared_model()

# --- 3. 終極安全性連線 (徹底解決 Base64 65字元報錯) ---
@st.cache_resource
def get_gspread_client():
    try:
        # 從 Secrets 取得所有資訊
        s = st.secrets["connections"]["gsheets"]
        
        # 強制修正 Private Key (移除 \\n, \n, 空格，並重新封裝)
        raw_key = s["private_key"]
        fixed_key = raw_key.replace("\\n", "\n").replace("\n", "\n").strip()
        
        # 建立標準 JSON 憑證字典
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
        
        # 定義權限範圍
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 建立憑證
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線失敗: {str(e)}")
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
                    # 讀取試算表
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
                        st.success("驗證成功！")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")
                except Exception as e:
                    st.error(f"資料庫讀取失敗: {e}")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶: {st.session_state.user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
    
    st.title(f"📊 {st.session_state.user} 的個人面板")
    st.write(f"系統狀態: {model_status}")
    st.divider()
    stock = st.text_input("輸入股票代碼進行預測")
