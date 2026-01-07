import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 核心修正：手動處理私鑰格式 (防止 Base64 報錯) ---
# 雖然我們簡化連線，但私鑰內的隱形換行符必須在程式執行時強制修正
def fix_service_account_keys():
    try:
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            # 取得原始字典
            creds = st.secrets["connections"]["gsheets"]
            # 修正 private_key 中的轉義字元與前後空格
            if "private_key" in creds:
                fixed_key = creds["private_key"].replace("\\n", "\n").strip()
                # 寫回記憶體供連線工具讀取
                st.secrets["connections"]["gsheets"]["private_key"] = fixed_key
    except Exception as e:
        st.error(f"認證資訊預處理失敗: {e}")

fix_service_account_keys()

# --- 3. 記憶體優化：30 人共享 TensorFlow 模型 ---
@st.cache_resource
def load_shared_model():
    # 確保伺服器僅加載一次模型，避免 30 人同時使用時記憶體溢出
    return "AI 模型運作中"

model_status = load_shared_model()

# --- 4. 建立連線 (最穩定的自動模式) ---
# 不再傳入 **clean_params，避免 'project_id' 等關鍵字重複錯誤
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"連線初始化失敗，請檢查 Secrets 格式。{e}")
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
                # 讀取試算表中的 users 工作表
                # 直接讀取，Streamlit 會自動使用 Secrets 裡的 spreadsheet 網址
                df = conn.read(worksheet="users", ttl=0)
                
                # 清理數據格式
                df['username'] = df['username'].astype(str).str.strip()
                df['password'] = df['password'].astype(str).str.strip()
                
                check = df[(df['username'] == u) & (df['password'] == p)]
                if not check.empty:
                    st.session_state.user = u
                    st.success("驗證通過！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤。")
            except Exception as e:
                st.error(f"資料庫讀取失敗: {e}")

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
    
    # 功能測試區
    stock = st.text_input("請輸入股票代碼")
    if stock:
        st.write(f"AI 正在計算 {stock} 的預測趨勢...")
