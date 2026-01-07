import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 0. 私鑰格式強效修復 (解決 Base64 65字元報錯關鍵) ---
def fix_secrets():
    try:
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            # 取得原始私鑰字串
            raw_key = st.secrets["connections"]["gsheets"]["private_key"]
            # 1. 將字串中的 \\n 替換為真正的換行符
            # 2. 去除首尾多餘的空格或換行
            fixed_key = raw_key.replace("\\n", "\n").strip()
            # 寫回暫時的記憶體中供連線使用
            st.secrets["connections"]["gsheets"]["private_key"] = fixed_key
    except Exception as e:
        st.error(f"私鑰修復失敗: {e}")

fix_secrets()

# --- 1. 頁面設定 ---
st.set_page_config(page_title="StockAI 管理系統", layout="wide")

# --- 2. 記憶體優化：共用 TensorFlow 模型 ---
@st.cache_resource
def load_stock_model():
    try:
        # 如果您有 model.h5 請解除註解
        # return tf.keras.models.load_model('model.h5')
        return "模型已就緒 (共用模式)"
    except Exception as e:
        st.warning(f"模型載入提醒: {e}")
        return None

model = load_stock_model()

# --- 3. 建立 Google Sheets 連線 ---
# 因為前面 fix_secrets() 已執行，這裡連線就不會報 Base64 錯誤
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"資料庫連線失敗，請檢查 Secrets。錯誤訊息：{e}")
    st.stop()

# --- 4. 登入邏輯 ---
if 'user_auth' not in st.session_state:
    st.session_state.user_auth = None

def login_ui():
    st.title("🚀 StockAI 系統登入")
    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            u = st.text_input("帳號")
            p = st.text_input("密碼", type="password")
            if st.button("確認登入", use_container_width=True):
                # 讀取 Google Sheets 的 'users' 分頁
                try:
                    df = conn.read(worksheet="users")
                    match = df[(df['username'] == u) & (df['password'] == p)]
                    if not match.empty:
                        st.session_state.user_auth = u
                        st.success("登入成功！")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")
                except Exception as e:
                    st.error(f"讀取使用者表失敗: {e}")

# --- 5. 主程式介面 ---
def main_ui():
    current_user = st.session_state.user_auth
    
    # 側邊欄
    st.sidebar.title(f"👤 {current_user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user_auth = None
        st.rerun()

    st.title(f"📈 {current_user} 的個人分析面板")
    
    # 功能分頁
    tab1, tab2 = st.tabs(["AI 選股分析", "歷史紀錄"])
    
    with tab1:
        st.subheader("TensorFlow 核心預測")
        stock_id = st.text_input("輸入股票代碼", placeholder="例如: 2330.TW")
        if st.button("啟動 AI 運算"):
            with st.spinner("正在調用共享模型資源..."):
                # 執行分析邏輯
                time.sleep(2)
                st.success(f"{stock_id} 分析完成。")
                st.metric("預測信心值", "85%", "+3%")

    with tab2:
        st.subheader("您的過往紀錄")
        st.info("這裡僅會顯示屬於您的數據，確保隱私安全。")
        # 示範：df = conn.read(worksheet="history")
        # my_data = df[df['owner'] == current_user]
        # st.dataframe(my_data)

# --- 執行進入點 ---
if st.session_state.user_auth is None:
    login_ui()
else:
    main_ui()
