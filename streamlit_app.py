import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 共享資源載入 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心修正：Base64 模數補齊 (徹底解決 65 字元報錯) ---
def clean_and_pad_key(raw_key):
    """移除雜質並自動補齊 Base64 填充字元"""
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 提取核心編碼內容
    core = raw_key.replace(header, "").replace(footer, "").replace("\\n", "").replace("\n", "").strip()
    
    # 只保留合法字元，過濾掉任何可能導致 65 字元的亂碼
    core = "".join(re.findall(r"[A-Za-z0-9\+/]", core))
    
    # 強制對齊：Base64 長度必須是 4 的倍數
    missing_padding = len(core) % 4
    if missing_padding:
        core += "=" * (4 - missing_padding)
    
    # 重新組合成 PEM 格式
    return f"{header}\n{core}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        s = st.secrets["connections"]["gsheets"]
        fixed_key = clean_and_pad_key(s["private_key"])
        
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
        st.error(f"安全性連線失敗：{str(e)}")
        return None

# --- 4. 登入介面 ---
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
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")
                except Exception as e:
                    st.error(f"資料表存取失敗。")

if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    st.title(f"📊 {st.session_state.user} 的個人面板")
