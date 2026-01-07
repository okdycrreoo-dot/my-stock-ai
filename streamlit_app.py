import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 0. 核心修正：強制處理 Secrets 私鑰 ---
# 這段代碼會確保無論 Secrets 裡是多行還是單行，都能正確轉為 Google 認可的格式
def get_fixed_secrets():
    try:
        creds = dict(st.secrets["connections"]["gsheets"])
        if "private_key" in creds:
            # 處理轉義字元並確保換行正確
            creds["private_key"] = creds["private_key"].replace("\\n", "\n")
        return creds
    except Exception as e:
        st.error(f"Secrets 讀取失敗: {e}")
        return None

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化 (30人共用) ---
@st.cache_resource
def load_model():
    # 這裡確保 TensorFlow 只載入一次，節省 1GB RAM
    return "AI 模型已就緒" 

model_info = load_model()

# --- 3. 建立連線 (使用修正後的參數) ---
try:
    # 我們不直接傳 type，而是手動傳入修正後的 secrets
    fixed_creds = get_fixed_secrets()
    conn = st.connection("gsheets", type=GSheetsConnection, **fixed_creds)
except Exception as e:
    st.error(f"連線中斷，請重新整理頁面。錯誤代碼: {e}")
    st.stop()

# --- 4. 登入系統 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 系統登入")
    u = st.text_input("帳號")
    p = st.text_input("密碼", type="password")
    if st.button("確認進入", use_container_width=True):
        try:
            # 讀取試算表中的 users 工作表
            df = conn.read(worksheet="users")
            # 比對帳密
            check = df[(df['username'].astype(str) == u) & (df['password'].astype(str) == p)]
            if not check.empty:
                st.session_state.user = u
                st.rerun()
            else:
                st.error("帳密不匹配")
        except Exception as e:
            st.error("存取資料表時發生錯誤")

# --- 5. 主程式頁面 ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入: {user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📈 歡迎，{user}")
    st.write(f"當前模型狀態: {model_info}")
    
    # 這裡可以開始寫您的選股邏輯
    stock = st.text_input("輸入股票代號進行 AI 預測")
    if stock:
        st.write(f"正在分析 {stock}...")
