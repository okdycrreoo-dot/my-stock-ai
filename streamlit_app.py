import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time
import re

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：預載模型 ---
@st.cache_resource
def load_shared_model():
    # 這裡未來可以替換成真實的模型載入代碼，例如 tf.keras.models.load_model()
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心連線函式 (解決 Unused bytes 與 欄位缺失問題) ---
@st.cache_resource
def get_stable_client():
    try:
        # 直接讀取整個 connections.gsheets 設定區塊轉為字典
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 關鍵：物理剔除私鑰中的非 ASCII 雜質 (解決 \xdab 報錯)
        if "private_key" in s_dict:
            # 丟掉所有無法辨識的二進位位元組，只保留標準文字
            clean_key = s_dict["private_key"].encode("ascii", "ignore").decode("utf-8")
            # 處理可能被誤轉義的換行符號
            s_dict["private_key"] = clean_key.replace("\\n", "\n").strip() + "\n"
        
        # 設定 Google API 權限範圍
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 使用字典內容直接建立憑證，這會自動匹配 project_id, client_x509_cert_url 等所有欄位
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
        
    except Exception as e:
        # 如果失敗，將具體錯誤顯示在頁面上供排查
        st.error(f"雲端連線失敗詳情: {str(e)}")
        return None

# --- 4. 登入邏輯 ---
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🚀 StockAI 登入系統")
    
    # 使用 Streamlit 表單組件
    with st.form("login_form"):
        u = st.text_input("帳號")
        p = st.text_input("密碼", type="password")
        submit = st.form_submit_button("進入系統", use_container_width=True)
        
        if submit:
            client = get_stable_client()
            if client:
                try:
                    # 從 Secrets 讀取試算表網址
                    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    # 開啟名為 'users' 的分頁
                    sheet = client.open_by_url(url).worksheet("users")
                    # 讀取所有資料並轉為 DataFrame
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    # 清洗資料：移除字串前後空格
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    # 比對帳密
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證通過！進入系統中...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤，請重新輸入")
                except Exception as e:
                    st.error("讀取失敗：請確保試算表中有 'users' 分頁，且包含 username 與 password 欄位。")
            else:
                st.info("提示：請檢查 Streamlit Secrets 設定是否完整（包含所有 client 欄位）。")

# --- 5. 主程式頁面 (登入後顯示) ---
if st.session_state.user is None:
    login()
else:
    # 側邊欄：顯示用戶資訊與登出按鈕
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    # 主畫面：分析面板
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("模型狀態", "運行中")
    with col2:
        st.metric("資料庫連線", "已連線")
        
    st.write(f"系統核心訊息：{model_status}")
    st.info("🎉 恭喜！你已經成功透過 Google Sheets 雲端資料庫完成身分驗證。")
