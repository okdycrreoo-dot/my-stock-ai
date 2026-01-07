import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# 頁面配置
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 1. 記憶體優化：共用模型 ---
@st.cache_resource
def load_model():
    # 這裡請確保您的 GitHub 倉庫中有 model.h5 檔案
    try:
        # return tf.keras.models.load_model('model.h5')
        return "模型載入成功 (模擬)"
    except:
        return None

model = load_model()

# --- 2. Google Sheets 連線 ---
# 確保 Secrets 已經設定好 [connections.gsheets]
conn = st.connection("gsheets", type=GSheetsConnection)

# --- 3. 登入系統 ---
if 'user_auth' not in st.session_state:
    st.session_state.user_auth = None

def check_login():
    st.title("🔐 StockAI 登入")
    with st.form("login_form"):
        user = st.text_input("帳號")
        pw = st.text_input("密碼", type="password")
        submit = st.form_submit_button("登入")
        
        if submit:
            # 從 Google Sheets 的 'users' 工作表讀取資料
            try:
                df = conn.read(worksheet="users")
                match = df[(df['username'] == user) & (df['password'] == pw)]
                if not match.empty:
                    st.session_state.user_auth = user
                    st.success("登入成功！")
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤")
            except Exception as e:
                st.error(f"資料庫連線失敗: {e}")

# --- 4. 主程式介面 ---
if st.session_state.user_auth is None:
    check_login()
else:
    user = st.session_state.user_auth
    st.sidebar.title(f"👤 {user}")
    if st.sidebar.button("登出"):
        st.session_state.user_auth = None
        st.rerun()

    st.title(f"📊 歡迎回來，{user}")
    
    # 互不干涉的核心：根據 user 篩選資料
    st.info("正在載入您的專屬數據...")
    
    # 範例：如果您的 'data' 工作表有一欄叫 'owner'
    # all_data = conn.read(worksheet="data")
    # user_data = all_data[all_data['owner'] == user]
    # st.dataframe(user_data)

    # 執行模型預測
    if st.button("啟動 AI 選股預測"):
        with st.spinner("AI 分析中..."):
            time.sleep(2) # 模擬運算
            st.success("分析完成！請查看下方結果。")
