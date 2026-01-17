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
# 工具章節：資料庫連線 (解決 NameError 的關鍵)
# ==========================================
@st.cache_resource
def init_db():
    """建立與 Google Sheets 的連線"""
    try:
        # 從 Streamlit Secrets 讀取設定
        info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(info, scopes=[
            'https://www.googleapis.com/auth/spreadsheets', 
            'https://www.googleapis.com/auth/drive'
        ])
        client = gspread.authorize(creds)
        # 開啟名為 "users" 的試算表及其同名工作表
        return client.open("users").worksheet("users")
    except Exception as e:
        st.error(f"❌ 資料庫連線失敗: {e}")
        return None

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
    # 1. 執行基礎樣式設定
    setup_page()
    
    # 2. 初始化登入狀態
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 3. 呼叫連線章節
    db = init_db() 
    
    if db is None:
        return

    # 4. 判斷頁面邏輯
    if not st.session_state["logged_in"]:
        # --- 入口頁面 (未登入) ---
        st.markdown("<h1 style='text-align: center;'>🔮 Oracle AI 入口頁面</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        with tab1:
            chapter_2_login(db)
        with tab2:
            chapter_1_registration(db)
            
    else:
        # --- 登入後：橫向緊湊佈局 (文字與按鈕 50% 縮小) ---
        # 1. 注入 CSS 縮小按鈕實體大小與字體
        st.markdown("""
            <style>
            div[data-testid="column"] { width: fit-content !important; flex: unset !important; }
            div[data-testid="stHorizontalBlock"] { gap: 10px; }
            /* 讓按鈕變小、字體變小、內距縮減 */
            .stButton > button {
                padding: 2px 10px !important;
                font-size: 12px !important;
                height: auto !important;
                min-height: 25px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # 2. 使用小比例欄位達成橫向緊靠
        c1, c2 = st.columns([0.1, 0.001], vertical_alignment="center")
        
        with c1:
            # 使用 span 標籤讓文字不換行，並縮小標題等級
            st.markdown(f"<h5 style='margin:0; white-space:nowrap;'>✅ 歡迎回來，{st.session_state['user']}！</h5>", unsafe_allow_html=True)
            
        with c2:
            if st.button("🚪 登出", key="main_logout"):
                st.session_state["logged_in"] = False
                st.rerun()

        # --- 第三章：監控清單管理預留區 ---
        st.markdown("---")
        st.subheader("📍 第三章：監控清單管理")
        st.info("導覽列已恢復橫向並縮小，準備進入功能開發。")

# 確保程式啟動
if __name__ == "__main__":
    main()




