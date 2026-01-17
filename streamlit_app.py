import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import json
import re
import time

# ==========================================
# 基礎配置：強制白色背景與 UI 修復
# ==========================================
def setup_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
        /* 修正灰色覆蓋層問題 */
        .stTabs [data-baseweb="tab-list"] { background-color: #FFFFFF; }
        p, label, h1, h2, h3 { color: #000000 !important; }
        input { background-color: #F0F2F6 !important; color: #000000 !important; }
        </style>
    """, unsafe_allow_html=True)

def is_alphanumeric(text):
    """英數檢查 (需求 1.5 & 2.5)"""
    return bool(re.match("^[a-zA-Z0-9]*$", text))

# ==========================================
# 第一章：帳號申請功能 (註冊物件)
# ==========================================

def reg_section(db_ws):
    # 1.1 設定帳號輸入框
    u = st.text_input("設定新帳號 (僅限英數)", key="reg_u_input")
    # 1.5 輸入限制
    if u and not is_alphanumeric(u):
        st.error("🚫 僅限英文或數字")
        
    # 1.2 設定密碼輸入框
    p = st.text_input("設定新密碼 (僅限英數)", type="password", key="reg_p_input")
    if p and not is_alphanumeric(p):
        st.error("🚫 僅限英文或數字")

    # 1.3 確認註冊按鈕
    if st.button("確認註冊帳號", key="reg_btn"):
        if not u or not p:
            st.warning("請填寫內容")
        elif not is_alphanumeric(u) or not is_alphanumeric(p):
            st.error("格式不符")
        else:
            # 1.4 確認重複邏輯
            users_list = db_ws.get_all_values()
            usernames = [row[0] for row in users_list] # 假設 A 欄是帳號
            if u in usernames:
                st.error(f"❌ 帳號 '{u}' 已存在")
            else:
                db_ws.append_row([u, p])
                st.success("🎉 註冊成功，請切換至登入頁面")

# ==========================================
# 第二章：帳號登入功能 (登入物件)
# ==========================================

def login_section(db_ws):
    # 2.1 帳號輸入框
    u = st.text_input("帳號", key="login_u_input")
    if u and not is_alphanumeric(u):
        st.error("🚫 僅限英文或數字")

    # 2.2 密碼輸入框
    p = st.text_input("密碼", type="password", key="login_p_input")
    if p and not is_alphanumeric(p):
        st.error("🚫 僅限英文或數字")

    # 2.3 確認登入按鈕
    if st.button("確認登入系統", key="login_btn"):
        if not u or not p:
            st.warning("請輸入帳號密碼")
        else:
            # 2.4 核對邏輯 (處理 '000000' 格式問題)
            users_list = db_ws.get_all_values()
            found = False
            for row in users_list:
                # 去除空白並強制轉字串比對
                db_u = str(row[0]).strip()
                db_p = str(row[1]).strip()
                if db_u == u and db_p == p:
                    found = True
                    break
            
            if found:
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.success("🎯 登入中...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ 核對失敗：帳號或密碼錯誤")

# ==========================================
# 入口頁面執行與對接 (Main Entrance)
# ==========================================

@st.cache_resource
def init_db():
    try:
        info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(info, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        return gspread.authorize(creds).open("users").worksheet("users")
    except:
        return None

def main():
    setup_theme()
    db = init_db()
    
    if db is None:
        st.error("資料庫連線中，請稍候...")
        return

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("🔮 Oracle AI 入口頁面")
        # 使用分頁區隔章節
        tab_login, tab_reg = st.tabs(["帳號登入", "帳號申請"])
        
        with tab_login:
            login_section(db)
            
        with tab_reg:
            reg_section(db)
    else:
        # 登入後的頁面預留 (第三章)
        st.title(f"歡迎回來, {st.session_state['user']}!")
        if st.button("登出"):
            st.session_state["logged_in"] = False
            st.rerun()

if __name__ == "__main__":
    main()
