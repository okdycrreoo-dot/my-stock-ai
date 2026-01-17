import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import json
import re

# ==========================================
# 基礎設定章節：強制白色主題與解鎖
# ==========================================
def setup_page():
    st.set_page_config(page_title="Oracle Login", layout="centered")
    st.markdown("""
        <style>
        /* 強制背景白色，並移除所有可能的灰色遮蓋層 */
        .stApp { background-color: #FFFFFF !important; }
        .stTabs [data-baseweb="tab-list"] { background-color: #FFFFFF !important; }
        p, label, h1, h2, h3 { color: #000000 !important; }
        /* 讓輸入框更明顯 */
        input { border: 1px solid #CCC !important; color: #000 !important; }
        </style>
    """, unsafe_allow_html=True)

def is_valid_format(text):
    """1.5 & 2.5 限制章節：僅限英數"""
    return bool(re.match("^[a-zA-Z0-9]*$", text))

# ==========================================
# 第一章：帳號申請功能 (註冊物件)
# ==========================================
def chapter_1_registration(db_ws):
    # 1.1 設定帳號輸入框
    u = st.text_input("設定新帳號", key="reg_u")
    if u and not is_valid_format(u):
        st.error("🚫 帳號僅能輸入英文或數字")

    # 1.2 設定密碼輸入框
    p = st.text_input("設定新密碼", type="password", key="reg_p")
    if p and not is_valid_format(p):
        st.error("🚫 密碼僅能輸入英文或數字")

    # 1.3 確認註冊按鈕
    if st.button("確認註冊並送出", key="reg_btn"):
        if u and p and is_valid_format(u) and is_valid_format(p):
            # 1.4 確認重複邏輯
            all_users = db_ws.col_values(1) # 只抓第一欄提升速度
            if u in all_users:
                st.error(f"❌ 帳號 '{u}' 已被使用")
            else:
                db_ws.append_row([u, p])
                st.success("🎉 註冊成功！請切換到登入分頁。")
        else:
            st.warning("請檢查輸入內容是否完整且格式正確。")

# ==========================================
# 第二章：帳號登入功能 (登入物件)
# ==========================================
def chapter_2_login(db_ws):
    # 2.1 帳號輸入框
    u = st.text_input("帳號", key="login_u")
    if u and not is_valid_format(u):
        st.error("🚫 請輸入英文或數字")

    # 2.2 密碼輸入框
    p = st.text_input("密碼", type="password", key="login_p")
    if p and not is_valid_format(p):
        st.error("🚫 請輸入英文或數字")

    # 2.3 確認登入按鈕
    if st.button("確認登入系統", key="login_btn"):
        if u and p:
            # 2.4 核對邏輯 (處理 000000 格式問題)
            data = db_ws.get_all_values()
            # 遍歷核對，強制轉字串解決 Google Sheets 格式問題
            match = any(str(row[0]).strip() == u and str(row[1]).strip() == p for row in data)
            
            if match:
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.rerun()
            else:
                st.error("❌ 帳號或密碼錯誤")

# ==========================================
# 核心執行入口章節 (The Main Entrance)
# ==========================================
def main():
    # 1. 基礎樣式設定
    setup_page()
    
    # 2. 初始化登入狀態 (必須放在最前面)
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 3. 呼叫資料庫連線 (請確認名稱是否為 init_db 或 get_database)
    # 如果你之前的連線函數叫 get_database，請把這裡改掉
    db = init_db() 
    
    if db is None:
        st.warning("資料庫連線中，請檢查 GCP Secrets 設定...")
        return

    # 4. 判斷頁面邏輯
    if not st.session_state["logged_in"]:
        # --- 第一、二章：登入註冊頁面 ---
        st.markdown("<h1 style='text-align: center;'>🔮 Oracle AI 入口頁面</h1>", unsafe_allow_html=True)
        
        # 使用簡單的分頁
        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        with tab1:
            chapter_2_login(db)
        with tab2:
            chapter_1_registration(db)
            
    else:
        # --- 登入後的並排佈局 (需求修正) ---
        # 建立兩個容器：左邊放歡迎文字，右邊放按鈕
        head_col1, head_col2 = st.columns([0.8, 0.2])
        
        with head_col1:
            # 用 markdown 顯示，避免 st.success 的大綠框擋住排版
            st.markdown(f"### ✅ 歡迎回來，{st.session_state['user']}！")
            
        with head_col2:
            # 對齊標題高度的登出按鈕
            st.write("##") # 補位調整
            if st.button("🚪 登出系統", key="main_logout"):
                st.session_state["logged_in"] = False
                st.rerun()

        # --- 第三章：監控清單管理物件預留區 ---
        st.markdown("---")
        st.subheader("📍 第三章：監控清單管理")
        st.write("目前清單功能正在對接中...")

# 執行點
if __name__ == "__main__":
    main()

