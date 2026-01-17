import streamlit as st
import re

# ==========================================
# 工具章節：格式檢查 (英文與數字限制)
# ==========================================
def validate_input(text):
    """檢查輸入是否僅包含英文字母與數字 (需求 1.5 & 2.5)"""
    if text == "": return True
    return bool(re.match("^[a-zA-Z0-9]*$", text))

# ==========================================
# 第一章：帳號申請功能 (Registration)
# ==========================================

def reg_input_username():
    """1.1 設定帳號輸入框"""
    u = st.text_input("設定新帳號 (僅限英數)", key="reg_u")
    if not validate_input(u):
        st.error("🚫 帳號含有非法字元，請僅使用英文或數字")
    return u

def reg_input_password():
    """1.2 設定密碼輸入框"""
    p = st.text_input("設定新密碼 (僅限英數)", type="password", key="reg_p")
    if not validate_input(p):
        st.error("🚫 密碼含有非法字元，請僅使用英文或數字")
    return p

def reg_check_duplicate(username, db_users):
    """1.4 確認帳號是否有重複"""
    # 這裡會接收來自資料庫的用戶清單進行比對
    is_duplicate = any(str(row['username']) == username for row in db_users)
    return is_duplicate

def reg_submit_button(u, p, db_users, db_connector):
    """1.3 確認註冊按鈕 (整合 1.4 邏輯)"""
    if st.button("確認註冊並建立帳號", key="btn_reg"):
        if not u or not p:
            st.warning("請完整填寫帳號與密碼")
        elif not validate_input(u) or not validate_input(p):
            st.error("請修正格式錯誤後再試")
        elif reg_check_duplicate(u, db_users):
            st.error(f"❌ 帳號 '{u}' 已被註冊，請換一個")
        else:
            # 執行寫入資料庫
            db_connector.append_row([u, p])
            st.success("🎉 註冊成功！請切換至登入頁面")

# ==========================================
# 第二章：帳號登入功能 (Login)
# ==========================================

def login_input_username():
    """2.1 帳號輸入框"""
    u = st.text_input("帳號", key="login_u")
    if not validate_input(u):
        st.error("🚫 格式錯誤：請輸入英文或數字")
    return u

def login_input_password():
    """2.2 密碼輸入框"""
    p = st.text_input("密碼", type="password", key="login_p")
    if not validate_input(p):
        st.error("🚫 格式錯誤：請輸入英文或數字")
    return p

def login_verify_credentials(u, p, db_users):
    """2.4 核對帳號密碼是否正確"""
    # 比對帳號密碼，並進行 strip() 去空格處理確保精準
    user_found = next((row for row in db_users if 
                       str(row['username']).strip() == u and 
                       str(row['password']).strip() == p), None)
    return user_found

def login_submit_button(u, p, db_users):
    """2.3 確認登入按鈕"""
    if st.button("確認登入系統", key="btn_login"):
        if login_verify_credentials(u, p, db_users):
            st.session_state["logged_in"] = True
            st.session_state["user"] = u
            st.success("🎯 驗證成功，正在進入終端...")
            st.rerun()
        else:
            st.error("❌ 登入失敗：帳號或密碼錯誤")

# ==========================================
# 執行入口 (展示如何拼接這些章節)
# ==========================================
def main_auth_page(db):
    st.title("🔮 Oracle AI 認證中心")
    tab_login, tab_reg = st.tabs(["登入系統", "帳號申請"])
    
    # 從資料庫獲取最新名單供 1.4 & 2.4 使用
    all_users = db["user_ws"].get_all_records()

    with tab_reg:
        u_reg = reg_input_username()
        p_reg = reg_input_password()
        reg_submit_button(u_reg, p_reg, all_users, db["user_ws"])

    with tab_login:
        u_log = login_input_username()
        p_log = login_input_password()
        login_submit_button(u_log, p_log, all_users)
