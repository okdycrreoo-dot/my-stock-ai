import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import tensorflow as tf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="StockAI 管理系統", layout="centered")

# --- 2. 記憶體優化：預載模型 ---
@st.cache_resource
def load_shared_model():
    return "AI 模型核心已就緒"

model_status = load_shared_model()

# --- 3. 核心連線函式 (解決編碼錯誤與欄位缺失問題) ---
@st.cache_resource
def get_stable_client():
    try:
        # 1. 讀取 Secrets 設定並轉為字典
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            st.error("找不到 Secrets 設定！請檢查 Streamlit Cloud 的 Secrets 區塊。")
            return None
            
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 2. 關鍵修正：物理剔除私鑰中的非 ASCII 雜質 (處理 \xdab 問題)
        if "private_key" in s_dict:
            # 丟掉所有非法位元組，只保留標準 ASCII 字元
            clean_key = s_dict["private_key"].encode("ascii", "ignore").decode("utf-8")
            # 處理可能出現的轉義換行
            s_dict["private_key"] = clean_key.replace("\\n", "\n").strip() + "\n"
        
        # 3. 設定 Google API 權限範圍
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 4. 建立憑證與授權
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
        
    except Exception as e:
        # 將具體錯誤顯示在紅框中
        st.error(f"雲端連線失敗詳情: {str(e)}")
        return None

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
            client = get_stable_client()
            if client:
                try:
                    # 從 Secrets 讀取試算表網址
                    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(url).worksheet("users")
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    # 統一清洗格式
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    # 帳密比對
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證通過！進入系統中...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤，請重新輸入")
                except Exception as e:
                    st.error(f"讀取失敗：請檢查試算表分頁名稱是否為 'users'。錯誤: {e}")
            else:
                st.info("提示：請檢查 Secrets 設定是否完整（建議包含所有 Google JSON 欄位）。")

# --- 5. 主程式分析面板 ---
if st.session_state.user is None:
    login()
else:
    st.sidebar.success(f"目前用戶：{st.session_state.user}")
    if st.sidebar.button("登出系統"):
        st.session_state.user = None
        st.rerun()
        
    st.title(f"📊 {st.session_state.user} 的分析面板")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("模型狀態", "運行中")
    with col2:
        st.metric("資料庫連線", "已連線")
        
    st.write(f"系統狀態：{model_status}")
    st.info("🎉 登入成功！您可以開始進行股票 AI 分析。")
