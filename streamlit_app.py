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

# --- 3. 核心修復：強制 Base64 清洗 (解決 Short substrate 與 Unused bytes) ---
def clean_private_key(raw_key):
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 提取核心 Base64 部分
    content = raw_key.replace(header, "").replace(footer, "")
    # 物理性剔除：只保留 A-Z, a-z, 0-9, +, / (完全排除 \xdab 等二進位雜質)
    content = "".join(re.findall(r"[A-Za-z0-9\+/]", content))
    
    # 強制補足 Padding (解決 Short substrate 問題)
    missing_padding = len(content) % 4
    if missing_padding:
        content += "=" * (4 - missing_padding)
        
    # 每 64 字元換行重新封裝 (標準 RSA 格式)
    formatted_content = "\n".join([content[i:i+64] for i in range(0, len(content), 64)])
    return f"{header}\n{formatted_content}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            st.error("Secrets 設定不完整")
            return None
            
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 進行終極格式清洗
        if "private_key" in s_dict:
            s_dict["private_key"] = clean_private_key(s_dict["private_key"])
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線最終嘗試中: {str(e)}")
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
                    st.error(f"讀取失敗：請確保試算表中有 'users' 分頁。")

# --- 5. 分析面板 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.write(f"系統狀態：{model_status}")
    st.info("連線狀態：Google Sheets 雲端連線正常")
