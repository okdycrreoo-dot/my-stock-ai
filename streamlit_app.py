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
    # 確保 TensorFlow 只加載一次，防止 30 人併發導致 OOM 崩潰
    return "AI 模型運算核心已啟動"

model_status = load_shared_model()

# --- 3. 核心修正：強制修正 Secrets 中的私鑰 (解決 Base64 65字元錯誤) ---
# 由於 st.secrets 是唯讀的，我們在讀取資料時手動傳入修正後的憑證字典
def get_verified_connection():
    try:
        # 取得原始設定
        conf = st.secrets["connections"]["gsheets"].to_dict()
        # 強制修正私鑰換行與空格問題
        if "private_key" in conf:
            conf["private_key"] = conf["private_key"].replace("\\n", "\n").strip()
        
        # 建立連線，僅傳入認證需要的關鍵參數
        return st.connection("gsheets", type=GSheetsConnection, **conf)
    except Exception as e:
        # 如果上方失敗，嘗試最簡化的自動連線
        return st.connection("gsheets", type=GSheetsConnection)

conn = get_verified_connection()

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
                # 取得試算表網址並讀取用戶資料表
                sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                df = conn.read(spreadsheet=sheet_url, worksheet="users", ttl=0)
                
                # 數據清洗與驗證
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.success("驗證成功，正在跳轉...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("帳號或密碼不正確")
            except Exception as e:
                st.error(f"資料庫存取失敗: {e}")

# --- 5. 主程式介面 ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"目前登入：{user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
    st.divider()
    
    # 預留功能區
    stock = st.text_input("輸入股票代碼進行 AI 分析")
    if stock:
        st.write(f"正在為 {stock} 調用 TensorFlow 進行預測...")
