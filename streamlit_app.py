import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型運算核心已啟動"

model_status = load_shared_model()

# --- 3. 核心修正：認證資訊封裝 (解決參數衝突與 Base64 錯誤) ---
def get_fixed_conn():
    try:
        # 1. 讀取 Secrets 並轉為一般字典
        raw_creds = st.secrets["connections"]["gsheets"].to_dict()
        
        # 2. 建立一個專門給 Google 認證用的內部字典
        # 將原本散落在外的參數全部包進 service_account_info
        service_account_info = {
            "type": "service_account",
            "project_id": raw_creds.get("project_id"),
            "private_key_id": raw_creds.get("private_key_id"),
            "private_key": raw_creds.get("private_key", "").replace("\\n", "\n").strip(),
            "client_email": raw_creds.get("client_email"),
            "client_id": raw_creds.get("client_id"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": raw_creds.get("client_x509_cert_url")
        }
        
        # 3. 建立連線：只傳入 service_account_info，不要傳入散裝的參數
        return st.connection(
            "gsheets", 
            type=GSheetsConnection, 
            service_account_info=service_account_info
        )
    except Exception as e:
        st.error(f"連線預處理失敗: {e}")
        return None

conn = get_fixed_conn()

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
            try:
                # 取得試算表網址
                sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                # 讀取 users 工作表 (ttl=0 確保即時驗證)
                df = conn.read(spreadsheet=sheet_url, worksheet="users", ttl=0)
                
                # 統一格式比對
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.success("登入成功！")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("帳號或密碼不正確")
            except Exception as e:
                st.error(f"資料庫讀取失敗，請確認分頁名稱為 'users'。錯誤: {e}")

# --- 5. 主程式介面 ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入用戶：{user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
    st.divider()
    
    stock = st.text_input("輸入股票代碼進行分析 (例: 2330)")
    if stock:
        st.write(f"正在分析 {stock} 的歷史數據...")
