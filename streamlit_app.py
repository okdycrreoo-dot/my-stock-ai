import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import json
import re

# ==========================================
# 系統設定章節：背景與基礎配置
# ==========================================
def setup_theme():
    """設定白色背景主題 (需求：背景白色)"""
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #000000; }
        p, label, h1, h2, h3 { color: #000000 !important; }
        .stButton>button { width: 100%; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 工具章節：英數限制檢查 (需求 1.5 & 2.5)
# ==========================================
def is_alphanumeric(text):
    """檢查輸入是否僅包含英文字母與數字"""
    return bool(re.match("^[a-zA-Z0-9]*$", text))

# ==========================================
# 第一章：帳號申請功能 (Registration)
# ==========================================

def reg_username_input():
    """1.1 設定帳號輸入框"""
    u = st.text_input("設定新帳號", key="reg_u", help="僅限輸入英文或數字")
    if not is_alphanumeric(u):
        st.error("⚠️ 帳號格式錯誤：請勿輸入符號或中文")
    return u

def reg_password_input():
    """1.2 設定密碼輸入框"""
    p = st.text_input("設定新密碼", type="password", key="reg_p")
    if not is_alphanumeric(p):
        st.error("⚠️ 密碼格式錯誤：請勿輸入符號或中文")
    return p

def reg_check_duplicate(u, users_data):
    """1.4 確認帳號是否有重複"""
    return any(str(row.get('username', '')).strip() == u for row in users_data)

def reg_submit_logic(u, p, db_ws):
    """1.3 確認註冊按鈕與執行邏輯"""
    if st.button("確認註冊帳號", key="btn_reg_submit"):
        # 再次抓取最新資料確認重複
        current_users = db_ws.get_all_records()
        if not u or not p:
            st.warning("請填寫帳號密碼")
        elif not is_alphanumeric(u) or not is_alphanumeric(p):
            st.error("請修正非英數格式")
        elif reg_check_duplicate(u, current_users):
            st.error(f"❌ 帳號 '{u}' 已存在，請更換")
        else:
            db_ws.append_row([u, p])
            st.success("🎉 註冊成功！現在可以切換到登入分頁了")

# ==========================================
# 第二章：帳號登入功能 (Login)
# ==========================================

def login_username_input():
    """2.1 帳號輸入框"""
    u = st.text_input("帳號", key="login_u")
    if not is_alphanumeric(u):
        st.error("⚠️ 格式不符：僅接受英文或數字")
    return u

def login_password_input():
    """2.2 密碼輸入框"""
    p = st.text_input("密碼", type="password", key="login_p")
    if not is_alphanumeric(p):
        st.error("⚠️ 格式不符：僅接受英文或數字")
    return p

def login_verify_logic(u, p, users_data):
    """2.3 & 2.4 確認登入按鈕與核對邏輯"""
    if st.button("確認登入系統", key="btn_login_submit"):
        # 尋找是否有匹配的帳號密碼
        found = next((row for row in users_data if 
                      str(row.get('username', '')).strip() == u and 
                      str(row.get('password', '')).strip() == p), None)
        if found:
            st.session_state["logged_in"] = True
            st.session_state["user"] = u
            st.success("🎯 驗證成功，正在登入...")
            st.rerun()
        else:
            st.error("❌ 核對失敗：帳號或密碼錯誤")

# ==========================================
# 資料庫連線章節 (Backend)
# ==========================================
@st.cache_resource
def get_database():
    try:
        info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(info, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        client = gspread.authorize(creds)
        sh = client.open("users")
        return sh.worksheet("users")
    except:
        return None

# ==========================================
# 執行主章節 (Main Entrance)
# ==========================================
def main():
    setup_theme()
    db_ws = get_database()
    
    if db_ws is None:
        st.error("資料庫連線失敗，請檢查權限設定")
        return

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 判斷登入狀態
    if not st.session_state["logged_in"]:
        st.title("🔮 Oracle AI 入口頁面")
        tab_login, tab_reg = st.tabs(["帳號登入", "帳號申請"])
        
        # 獲取基礎數據供比對
        users_data = db_ws.get_all_records()

        with tab_reg:
            u_r = reg_username_input()
            p_r = reg_password_input()
            reg_submit_logic(u_r, p_r, db_ws)

        with tab_login:
            u_l = login_username_input()
            p_l = login_password_input()
            login_verify_logic(u_l, p_l, users_data)
    else:
        # --- 登入後的第三章預留位置 ---
        st.title(f"歡迎, {st.session_state['user']}!")
        st.info("這裡是登入後的設計區塊。")
        if st.button("登出"):
            st.session_state["logged_in"] = False
            st.rerun()

if __name__ == "__main__":
    main()
