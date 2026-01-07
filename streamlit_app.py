import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 1. 頁面基本配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="wide")

# --- 2. 記憶體優化：30 人共用一個 TensorFlow 模型 ---
# 使用 cache_resource 避免重複載入導致 1GB RAM 崩潰
@st.cache_resource
def get_model():
    try:
        # 如果您有模型檔案，請將下行註解拿掉
        # return tf.keras.models.load_model('model.h5')
        return "模型已就緒"
    except Exception as e:
        return f"模型載入提醒: {e}"

model_status = get_model()

# --- 3. 建立 Google Sheets 連線 ---
# 直接連接，不需額外轉換，Streamlit 會自動解析 Secrets 裡的 \n
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"Google Sheets 連線失敗，請檢查 Secrets 格式。錯誤訊息: {e}")
    st.stop()

# --- 4. 登入狀態管理 ---
if 'user_auth' not in st.session_state:
    st.session_state.user_auth = None

# --- 登入畫面 ---
def show_login():
    st.title("🚀 StockAI 系統登入")
    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            u = st.text_input("帳號 (Username)")
            p = st.text_input("密碼 (Password)", type="password")
            if st.button("確認登入", use_container_width=True):
                try:
                    # 讀取 users 分頁進行驗證
                    df = conn.read(worksheet="users")
                    # 檢查是否有匹配的帳密
                    user_match = df[(df['username'].astype(str) == u) & (df['password'].astype(str) == p)]
                    
                    if not user_match.empty:
                        st.session_state.user_auth = u
                        st.success("驗證成功，正在進入系統...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤，請重新輸入。")
                except Exception as e:
                    st.error(f"無法存取使用者清單: {e}")

# --- 主程式畫面 (登入後) ---
def show_main():
    user = st.session_state.user_auth
    
    # 側邊欄控制
    st.sidebar.title(f"👤 使用者: {user}")
    if st.sidebar.button("登出"):
        st.session_state.user_auth = None
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.write(f"系統狀態: {model_status}")

    # 主功能區
    st.title(f"📈 歡迎回來，{user}")
    
    tab1, tab2 = st.tabs(["AI 選股分析", "個人歷史紀錄"])
    
    with tab1:
        st.subheader("TensorFlow 核心預測")
        stock_id = st.text_input("輸入股票代碼", placeholder="例如: 2330.TW")
        if st.button("執行 AI 運算"):
            with st.spinner("AI 正在分析大數據..."):
                # 這裡放入您的預測邏輯
                time.sleep(2)
                st.success(f"{stock_id} 分析完成")
                st.metric(label="預測趨勢", value="看多", delta="85% 信心度")

    with tab2:
        st.subheader("我的操作紀錄")
        st.info("這裡僅顯示您個人的分析歷史。")
        # 範例：篩選屬於該使用者的資料列
        # all_logs = conn.read(worksheet="history")
        # my_logs = all_logs[all_logs['owner'] == user]
        # st.dataframe(my_logs)

# --- 程式執行邏輯 ---
if st.session_state.user_auth is None:
    show_login()
else:
    show_main()
