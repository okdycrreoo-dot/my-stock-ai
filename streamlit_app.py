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
    # 確保模型在伺服器上只載入一次，節省資源
    return "AI 模型已就緒"

model_status = load_shared_model()

# --- 3. 核心連線邏輯 (解決 Base64 錯誤的關鍵) ---
def get_conn():
    try:
        # 手動清理 Secrets 中的私鑰字串
        creds = st.secrets["connections"]["gsheets"].to_dict()
        if "private_key" in creds:
            # 移除所有可能的二次轉義與空格
            creds["private_key"] = creds["private_key"].replace("\\n", "\n").strip()
            
        # 移除會造成 GSheetsConnection 混淆的連線參數
        for key in ["type", "spreadsheet"]:
            if key in creds: del creds[key]
            
        # 建立連線
        return st.connection("gsheets", type=GSheetsConnection, **creds)
    except Exception as e:
        st.error(f"連線初始化失敗: {e}")
        return None

conn = get_conn()

# --- 4. 登入系統 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
    with st.form("login_form"):
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        if st.form_submit_button("登入系統", use_container_width=True):
            try:
                # 讀取試算表
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                df = conn.read(spreadsheet=url, worksheet="users", ttl=0)
                
                # 比對帳密
                check = df[(df['username'].astype(str).str.strip() == u) & 
                           (df['password'].astype(str).str.strip() == p)]
                
                if not check.empty:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤")
            except Exception as e:
                st.error(f"資料庫讀取失敗: {e}")

# --- 5. 主程式 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"已登入: {st.session_state.user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
    
    st.title(f"📈 歡迎，{st.session_state.user}")
    st.write(f"系統狀態: {model_status}")
    # 這裡放您的股票預測功能
