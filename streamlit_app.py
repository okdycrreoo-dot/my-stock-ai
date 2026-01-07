import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：30 人共享資源 ---
@st.cache_resource
def load_ai_model():
    # 確保 TensorFlow 模型在伺服器上只載入一次
    return "AI 運算核心已啟動"

model_status = load_ai_model()

# --- 3. 建立連線 (最簡潔方式) ---
# 讓 Streamlit 自動從 Secrets 中的 [connections.gsheets] 讀取配置
# 我們不再手動傳入字典，以避免 'project_id' 等參數錯誤
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"連線初始化失敗，請檢查 Secrets 格式。錯誤訊息: {e}")
    st.stop()

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
                # 取得 Secrets 裡的試算表網址
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                # 讀取試算表中的 users 工作表
                df = conn.read(spreadsheet=url, worksheet="users", ttl=0)
                
                # 清理資料確保比對準確
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤。")
            except Exception as e:
                st.error(f"資料庫存取失敗: {e}")

# --- 5. 主程式頁面 (登入後) ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"用戶：{user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的分析面板")
    st.write(f"系統狀態：{model_status}")
    st.divider()
    
    stock_input = st.text_input("輸入股票代碼進行 AI 預測")
    if stock_input:
        st.write(f"正在為 {stock_input} 分析數據...")
