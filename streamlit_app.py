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
    # 執行基礎設定 (白色背景、英數限制提示等)
    setup_page()
    
    # 執行資料庫連線章節
    db = init_db() 
    
    if db is None:
        return

    # 初始化登入狀態
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 判斷登入狀態並顯示對應章節
    if not st.session_state["logged_in"]:
        # --- 顯示登入/註冊頁面 (第一、二章) ---
        st.markdown("<h1 style='text-align: center;'>🔮 Oracle AI 入口頁面</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        
        with tab1:
            chapter_2_login(db)
        with tab2:
            chapter_1_registration(db)
            
    else:
        # --- 顯示登入後的並排佈局 (需求：文字後方跟隨登出按鈕) ---
        # 建立兩個水平欄位，第一欄放文字，第二欄放按鈕
        col_msg, col_logout = st.columns([0.85, 0.15])
        
        with col_msg:
            # 使用 markdown 展示標題，確保與按鈕處於同一行
            st.markdown(f"### ✅ 歡迎回來，{st.session_state['user']}！")
            
        with col_logout:
            # 增加間距對齊文字高度
            st.write("##") 
            if st.button("登出系統", key="main_logout"):
                st.session_state["logged_in"] = False
                # 清除狀態並重整
                st.rerun()

        # --- 以下開始進入第三章預留位置 ---
        st.markdown("---")
        st.subheader("📍 第三章：監控清單管理")
        st.info("此區塊將放置：自選股表格、新增輸入框、刪除按鈕等物件。")

# 程式啟動點
if __name__ == "__main__":
    main()


