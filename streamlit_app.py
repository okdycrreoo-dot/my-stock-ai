import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 核心修正：手動清洗 Secrets (排除參數衝突) ---
def get_clean_params():
    try:
        # 抓取 Secrets 設定
        creds = st.secrets["connections"]["gsheets"].to_dict()
        
        # 修正私鑰換行符號
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace("\\n", "\n").strip()
        
        # 移除不屬於連線初始化用的參數
        # 'type' 與 st.connection 的參數重複
        # 'spreadsheet' 是讀取時才用的，不是連線時用的
        keys_to_remove = ["type", "spreadsheet"]
        for key in keys_to_remove:
            if key in creds:
                del creds[key]
            
        return creds
    except Exception as e:
        st.error(f"Secrets 讀取異常: {e}")
        return None

# --- 3. 記憶體優化：30 人共享資源 ---
@st.cache_resource
def load_ai_model():
    return "AI 運算核心已啟動"

model_status = load_ai_model()

# --- 4. 建立連線 ---
try:
    clean_params = get_clean_params()
    # 建立連線，這裡僅傳入認證所需的參數
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
                # 這裡使用 secrets 裡的網址讀取
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                df = conn.read(spreadsheet=url, worksheet="users", ttl=0)
                
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤。")
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
    st.divider()
    
    stock_input = st.text_input("請輸入股票代碼 (例: 2330)")
    if stock_input:
        st.write(f"正在為 {stock_input} 進行 AI 預測...")
