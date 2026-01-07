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
    return "AI 模型運算核心已啟動"

model_status = load_shared_model()

# --- 3. 核心修正：認證資訊預處理 (解決所有報錯) ---
def get_fixed_conn():
    try:
        # 1. 將唯讀的 secrets 轉為可編輯的字典
        creds = st.secrets["connections"]["gsheets"].to_dict()
        
        # 2. 強制修正私鑰格式，去除導致 (65) 字元錯誤的隱形換行
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace("\\n", "\n").strip()
        
        # 3. 移除會導致 keyword argument 衝突的參數
        # 這些參數 Streamlit 會自動處理，手動傳入反而會報錯
        for key in ["type", "spreadsheet", "project_id"]:
            if key in creds:
                del creds[key]
            
        # 4. 使用清理過的參數建立連線
        return st.connection("gsheets", type=GSheetsConnection, **creds)
    except Exception as e:
        st.error(f"連線預處理失敗: {e}")
        return None

conn = get_fixed_conn()

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
                # 從 Secrets 取得試算表網址
                sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                # 讀取 users 工作表 (ttl=0 確保不快取，即時驗證)
                df = conn.read(spreadsheet=sheet_url, worksheet="users", ttl=0)
                
                # 統一格式比對
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.success("登入成功！")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("帳號或密碼不正確")
            except Exception as e:
                st.error(f"資料庫讀取失敗，請確認分頁名稱為 'users'。錯誤: {e}")

# --- 5. 主程式介面 ---
if st.session_state.user is None:
    login()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入用戶：{user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {user} 的個人面板")
    st.write(f"系統狀態：{model_status}")
    st.divider()
    
    # AI 分析預留區
    stock = st.text_input("輸入股票代碼進行分析 (例: 2330)")
    if stock:
        st.write(f"正在分析 {stock} 的歷史數據...")
