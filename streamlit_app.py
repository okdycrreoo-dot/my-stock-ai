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
# 執行入口章節 - 登入後狀態調整
# ==========================================

# ... (前面 setup_page, chapter_1, chapter_2 保持不變) ...

def main():
    setup_page()
    db = init_db()
    
    # ... (資料庫連線檢查與 session_state 初始化) ...

    if not st.session_state["logged_in"]:
        # 顯示登入/註冊分頁 (第一、二章)
        st.title("🔮 Oracle AI 入口頁面")
        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        with tab1: chapter_2_login(db)
        with tab2: chapter_1_registration(db)
    else:
        # --- 登入後的佈局調整 ---
        # 使用 columns 讓文字與按鈕並排
        # [4, 1] 代表左邊佔 4 份寬度，右邊按鈕佔 1 份，這樣按鈕會靠右且跟在後面
        col_text, col_btn = st.columns([4, 1])
        
        with col_text:
            st.success(f"歡迎回來，{st.session_state['user']}")
            
        with col_btn:
            # 為了美觀，我們加一點空間讓按鈕對齊文字高度
            st.write("") 
            if st.button("登出", key="logout_btn"):
                st.session_state["logged_in"] = False
                st.rerun()
        
        # --- 接下來可以開始設計第三章的內容 ---
        st.markdown("---")
        st.write("📍 這裡將開始放置第三章：股票監控清單管理物件")

if __name__ == "__main__":
    main()

