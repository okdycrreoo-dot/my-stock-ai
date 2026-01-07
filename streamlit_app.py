import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：方案 A 的私鑰處理函式 ---
def get_pure_private_key(raw_key):
    """
    針對 Streamlit Secrets 可能產生的 Unused bytes (\xdab) 進行物理剔除
    """
    try:
        # 1. 物理剔除非 ASCII 字元 (徹底解決 \xdab 問題)
        # encode('ascii', 'ignore') 會直接丟掉無法辨識的二進位位元組
        clean_key = raw_key.encode("ascii", "ignore").decode("utf-8")
        
        # 2. 處理可能被誤轉義的斜槓
        clean_key = clean_key.replace("\\n", "\n")
        
        # 3. 確保前後沒有多餘空格
        return clean_key.strip() + "\n"
    except Exception as e:
        st.error(f"私鑰處理出錯: {e}")
        return raw_key

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用優化後的私鑰處理
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
        # 這裡會顯示具體的錯誤，方便我們排查
        st.error(f"連線失敗詳情: {str(e)}")
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
                    
                    # 清洗資料庫中的空白
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
                    st.error(f"讀取失敗：請檢查分頁 'users' 是否存在且格式正確。")
            else:
                st.error("無法建立雲端連線，請檢查 Secrets 設定。")

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
    st.info("您已成功連接 Google Sheets 資料庫！")
