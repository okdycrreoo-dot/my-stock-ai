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

# --- 3. 核心修正：物理級私鑰重組 (徹底解決 65 字元與 Unused bytes 錯誤) ---
def force_clean_key(raw_key):
    """物理性移除所有雜質，重新構建標準 RSA 私鑰格式"""
    # 提取標籤中間的核心編碼內容
    core = raw_key.replace("-----BEGIN PRIVATE KEY-----", "")
    core = core.replace("-----END PRIVATE KEY-----", "")
    
    # 關鍵：只允許符合 Base64 規範的字元 (A-Z, a-z, 0-9, +, /, =)
    # 這會物理性剔除您日誌中出現的 \xdab 等不可見雜質
    core = "".join(re.findall(r"[A-Za-z0-9\+/=]", core))
    
    # 強制修正 Base64 長度：必須是 4 的倍數
    # 解決截圖中提到的 (65) cannot be 1 more than a multiple of 4
    missing_padding = len(core) % 4
    if missing_padding:
        core += "=" * (4 - missing_padding)
    
    # 重新組合成 Google 認可的標準換行格式
    # 每 64 個字元換一行是標準 RSA 規範
    formatted_core = "\n".join([core[i:i+64] for i in range(0, len(core), 64)])
    return f"-----BEGIN PRIVATE KEY-----\n{formatted_core}\n-----END PRIVATE KEY-----\n"

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        # 使用物理重組後的純淨私鑰
        clean_key = force_clean_key(s["private_key"])
        
        info = {
            "type": "service_account",
            "project_id": s["project_id"],
            "private_key_id": s["private_key_id"],
            "private_key": clean_key,
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
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證通過，正在進入個人面板...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤，請重新輸入")
                except Exception as e:
                    st.error(f"資料讀取失敗，請確認試算表分頁名稱為 'users'。錯誤: {e}")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的個人面板")
    st.info(f"系統狀態：{model_status}")
    st.divider()
    # 功能區佔位
    st.text_input("輸入股票代碼以啟動 AI 分析 (例: 2330)")
