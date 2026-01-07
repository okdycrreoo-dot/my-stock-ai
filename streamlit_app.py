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

# --- 3. 核心連線函式 (終極 RSA 格式修復版) ---
@st.cache_resource
def get_stable_client():
    try:
        # 1. 讀取 Secrets 設定
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            st.error("找不到 Secrets 設定！請檢查 Streamlit Cloud 的 Secrets 區塊。")
            return None
            
        s_dict = dict(st.secrets["connections"]["gsheets"])
        
        # 2. 強力修復私鑰格式 (解決 asn1Spec 與 Unused bytes 問題)
        if "private_key" in s_dict:
            raw_key = s_dict["private_key"]
            
            # 處理轉義換行
            fixed_key = raw_key.replace("\\n", "\n")
            
            header = "-----BEGIN PRIVATE KEY-----"
            footer = "-----END PRIVATE KEY-----"
            
            if header in fixed_key and footer in fixed_key:
                # 提取中間的核心內容，移除所有空格、換行、Tab
                core = fixed_key.split(header)[1].split(footer)[0]
                # 物理剔除非 ASCII 雜質並刪除所有空白字元
                core_clean = "".join(re.findall(r"[A-Za-z0-9\+/=]", core))
                
                # 依照標準 RSA 格式：每 64 字元換一行重新組裝
                formatted_core = "\n".join([core_clean[i:i+64] for i in range(0, len(core_clean), 64)])
                s_dict["private_key"] = f"{header}\n{formatted_core}\n{footer}\n"
        
        # 3. 設定 Google API 權限範圍
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 4. 建立憑證
        creds = Credentials.from_service_account_info(s_dict, scopes=scopes)
        return gspread.authorize(creds)
        
    except Exception as e:
        st.error(f"安全性連線最終嘗試中: {str(e)}")
        return None

# --- 4. 登入系統邏輯 ---
import re # 確保匯入正規表達式

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
                    url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    sheet = client.open_by_url(url).worksheet("users")
                    df = pd.DataFrame(sheet.get_all_records())
                    
                    df['username'] = df['username'].astype(str).str.strip()
                    df['password'] = df['password'].astype(str).str.strip()
                    
                    check = df[(df['username'] == u) & (df['password'] == p)]
                    
                    if not check.empty:
                        st.session_state.user = u
                        st.success("驗證通過！")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤")
                except Exception as e:
                    st.error(f"讀取失敗：請檢查試算表分頁名稱是否為 'users'。錯誤: {e}")
            else:
                st.info("提示：請檢查 Secrets 設定是否完整。")

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
    st.info("🎉 恭喜！您已成功連線雲端資料庫。")
