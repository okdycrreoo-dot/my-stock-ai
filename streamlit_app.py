import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import tensorflow as tf
import time

# --- 頁面設定 ---
st.set_page_config(page_title="StockAI 投資管理系統", layout="wide")

# --- 1. 記憶體優化：共用 TensorFlow 模型 ---
# 使用 cache_resource 確保 30 人共用同一個模型，避免記憶體溢出 (OOM)
@st.cache_resource
def load_stock_model():
    try:
        # 替換為您的模型路徑，例如 'model.h5'
        # model = tf.keras.models.load_model('your_model.h5')
        # return model
        return "模型載入成功 (模擬)" 
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None

model = load_stock_model()

# --- 2. 建立 Google Sheets 連線 ---
conn = st.connection("gsheets", type=GSheetsConnection)

# 讀取使用者資料表 (假設工作表名稱為 'users')
def get_user_data():
    return conn.read(worksheet="users", ttl=5) # ttl=5 表示每 5 秒快取過期

# --- 3. 登入邏輯 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def login():
    st.title("🚀 StockAI 系統登入")
    
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            username = st.text_input("帳號")
            password = st.text_input("密碼", type="password")
            login_btn = st.button("確認登入", use_container_width=True)

    if login_btn:
        user_df = get_user_data()
        # 驗證帳號密碼
        user_match = user_df[(user_df['username'] == username) & (user_df['password'] == password)]
        
        if not user_match.empty:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"歡迎回來，{username}！")
            time.sleep(1)
            st.rerun()
        else:
            st.error("帳號或密碼不正確，請重新檢查。")

# --- 4. 主程式內容 ---
def main_app():
    user = st.session_state.username
    
    # 側邊欄
    st.sidebar.title("控制面板")
    st.sidebar.write(f"當前使用者：**{user}**")
    if st.sidebar.button("登出"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.title(f"📈 {user} 的專屬選股工作區")
    
    # 功能區塊
    tab1, tab2 = st.tabs(["AI 選股預測", "個人操作紀錄"])
    
    with tab1:
        st.subheader("TensorFlow AI 預測模型")
        stock_code = st.text_input("輸入股票代號 (例如: 2330.TW)")
        
        if st.button("開始分析"):
            with st.spinner("AI 運算中..."):
                # 這裡執行您的 TensorFlow 預測邏輯
                # result = model.predict(data)
                time.sleep(2)
                st.success(f"股票 {stock_code} 分析完成！預測結果：看多 (模擬)")
                
                # 將結果存回試算表 (假設有另一個工作表叫 'logs')
                new_log = pd.DataFrame([{"user": user, "stock": stock_code, "action": "分析", "time": time.ctime()}])
                # 注意：st-gsheets-connection 更新資料通常需要先讀取再寫入，或使用其 update 方法
                # st.write("紀錄已同步至 Google Sheets")

    with tab2:
        st.subheader("您的歷史紀錄")
        # 這裡示範如何過濾「只顯示該使用者」的資料，達成互不干涉
        # all_logs = conn.read(worksheet="logs")
        # my_logs = all_logs[all_logs['user'] == user]
        # st.dataframe(my_logs)
        st.info("這裡將顯示您過去的選股分析紀錄。")

# --- 執行進入點 ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login()
    else:
        main_app()
