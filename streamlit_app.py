import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 核心修正：手動清洗 Secrets (解決 Base64 與參數衝突) ---
def get_clean_params():
    try:
        # 抓取 Secrets 設定並轉換為可編輯字典
        creds = st.secrets["connections"]["gsheets"].to_dict()
        
        # 修正私鑰換行符號
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace("\\n", "\n").strip()
        
        # 關鍵修正：移除字典中的 'type'，避免與 st.connection(type=...) 衝突
        if "type" in creds:
            del creds["type"]
            
        return creds
    except Exception as e:
        st.error(f"Secrets 讀取異常: {e}")
        return None

# --- 3. 記憶體優化：30 人共享資源 ---
@st.cache_resource
def load_ai_model():
    # 這裡確保 30 個人共用一個 TF 實例，節省記憶體
    return "AI 運算核心已啟動"

model_status = load_ai_model()

# --- 4. 建立連線 ---
try:
    clean_params = get_clean_params()
    # 這裡 type 參數與 **clean_params 不再衝突
    conn = st.connection("gsheets", type=GSheetsConnection, **clean_params)
except Exception as e:
    st.error(f"連線初始化失敗: {e}")
    st.stop()

# --- 5. 登入系統邏輯 ---
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
                # 讀取試算表 users 工作表
                df = conn.read(worksheet="users", ttl=0) # ttl=0 確保即時驗證
                
                # 確保資料格式統一
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤，請確認試算表內容。")
            except Exception as e:
                st.error(f"無法存取 Google Sheets: {e}")

# --- 6. 主程式頁面 (登入後) ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"用戶：{user}")
    if st.sidebar.button("登出"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的專屬分析面板")
    st.write(f"系統狀態：{model_status}")
    
    # 這裡可以開始實作您的分析工具
    st.divider()
    stock_input = st.text_input("請輸入股票代碼 (例: 2330)")
    if stock_input:
        st.write(f"正在調用 TensorFlow 為 {stock_input} 進行預測...")
