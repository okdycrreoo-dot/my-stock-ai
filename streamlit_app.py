import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 核心修復：終極私鑰清洗函式 ---
def get_pure_private_key(raw_key):
    """
    徹底過濾非合法 Base64 字元，解決 Short substrate 與 Unused bytes 問題
    """
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    # 提取核心內容 (移除標頭、標尾、轉義換行)
    core = raw_key.replace(header, "").replace(footer, "").replace("\\n", "").replace("\n", "").strip()
    
    # 關鍵：只保留 Base64 合法字元集 (A-Z, a-z, 0-9, +, /)
    # 這會物理性剔除導致報錯的 \xdab 等二進位雜質
    core = "".join(re.findall(r"[A-Za-z0-9\+/]", core))
    
    # 強制補齊填充字元 '=' 至 4 的倍數 (解決 Short substrate 錯誤)
    missing_padding = len(core) % 4
    if missing_padding:
        core += "=" * (4 - missing_padding)
    
    # 按照 Google 標準格式：每 64 個字元換一行重新排版
    formatted_body = "\n".join([core[i:i+64] for i in range(0, len(core), 64)])
    
    return f"{header}\n{formatted_body}\n{footer}\n"

@st.cache_resource
def get_stable_client():
    try:
        # 讀取並轉為字典
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 使用修復函式處理私鑰
        if "private_key" in s_dict:
            s_dict["private_key"] = get_pure_private_key(s_dict["private_key"])
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"安全性連線最終嘗試中: {str(e)}")
        return None

# --- 3. 登入邏輯 ---
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
                    st.error(f"資料讀取失敗，請確認分頁 'users' 存在。")

# --- 4. 主程式頁面 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.info("雲端連線狀態：正常 ✅")
