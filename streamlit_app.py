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
    try:
        info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(info, scopes=[
            'https://www.googleapis.com/auth/spreadsheets', 
            'https://www.googleapis.com/auth/drive'
        ])
        client = gspread.authorize(creds)
        spreadsheet = client.open("users") # 打開試算表檔案
        return {
            "users": spreadsheet.worksheet("users"),
            "watchlist": spreadsheet.worksheet("watchlist"),
            "predictions": spreadsheet.worksheet("predictions")
        }
    except Exception as e:
        st.error(f"❌ 資料庫分頁連線失敗: {e}")
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
    setup_page()
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    db_dict = init_db() 
    if db_dict is None:
        return

    if not st.session_state["logged_in"]:
        # --- 入口頁面 (未登入) ---
        st.markdown("<h1 style='text-align: center;'>🔮 Oracle AI 入口頁面</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        with tab1:
            chapter_2_login(db_dict["users"]) # 傳入 users 分頁
        with tab2:
            chapter_1_registration(db_dict["users"])
            
    else:
        # --- 登入後：導覽列 ---
        st.markdown("""
            <style>
            div[data-testid="column"] { width: fit-content !important; flex: unset !important; }
            div[data-testid="stHorizontalBlock"] { gap: 10px; }
            .stButton > button { padding: 2px 10px !important; font-size: 12px !important; min-height: 25px !important; }
            </style>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([0.1, 0.03], vertical_alignment="center")
        with c1:
            st.markdown(f"<h5 style='margin:0; white-space:nowrap;'>✅ 歡迎回來，{st.session_state['user']}！</h5>", unsafe_allow_html=True)
        with c2:
            if st.button("🚪 登出", key="main_logout"):
                st.session_state["logged_in"] = False
                st.rerun()

        st.markdown("---")

        # 【核心修正】在這裡呼叫第三章，縮放按鈕才會出現！
        chapter_3_watchlist_management(
            db_dict["users"], 
            db_dict["watchlist"], 
            db_dict["predictions"]
        )

# ==========================================
# 第三章：監控清單管理功能 (Control Panel)
# ==========================================

def chapter_3_watchlist_management(db_ws, watchlist_ws, predictions_ws):
    user_name = st.session_state["user"]
    
    # 1. 取得目前使用者的自選清單
    try:
        all_watch = watchlist_ws.get_all_values()
        # 假設 A 欄是 User, B 欄是股票代號
        user_stocks = [row[1] for row in all_watch if len(row) > 1 and row[0] == user_name]
    except:
        user_stocks = []
    
    stock_count = len(user_stocks)

    # --- 3.1 整個功能都裝進縮放按鈕 ---
    with st.expander("🛠️ 開啟股票控制台", expanded=False):
        
        # 3.2 上半部：新增功能佈局
        st.write(f"### 📥 新增自選股 ({stock_count}/30)")
        col_input, col_add = st.columns([3, 1])
        
        with col_input:
            new_stock = st.text_input("輸入股票代號 (英數)", key="new_stock_input").strip().upper()
        
        with col_add:
            st.write("##") # 對齊
            add_btn = st.button("確認新增", key="add_stock_btn")
            
        # 3.3 新增邏輯處理 (您要求的邏輯都在這)
        if add_btn:
            if not new_stock:
                st.warning("⚠️ 請先輸入代號")
            elif not is_valid_format(new_stock):
                st.error("🚫 格式錯誤：僅限輸入英文或數字")
            elif stock_count >= 30:
                st.warning("⚠️ 已達上限：最多只能 30 筆自選股")
            elif any(s.startswith(new_stock) for s in user_stocks):
                st.info("💡 提醒：此股票已在清單中")
            else:
                # --- 自動比對市場尾數邏輯 (.TW / .TWO) ---
                # 這裡目前以簡單判斷示範：一般 4 位代號且不以 '8' 或 '6' 開頭多為上市
                # 實際建議對接市場名單，這裡先預設處理邏輯：
                if len(new_stock) == 4 and new_stock[0] in ['2', '3']:
                    suffix = ".TW"
                else:
                    suffix = ".TWO"
                
                full_code = f"{new_stock}{suffix}"
                
                # 寫入試算表 (User, Full_Code)
                watchlist_ws.append_row([user_name, full_code])
                st.success(f"✅ {full_code} 已加入清單")
                st.rerun()

        st.markdown("---")
            
        # 3.4 下半部：自選股清單顯示 (下拉選單形式)
        st.write("### 📋 監控清單管理")
        if not user_stocks:
            st.info("目前清單中沒有股票")
        else:
            c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
            
            with c1:
                selected_stock = st.selectbox("選擇要操作的股票", options=user_stocks, key="stock_selector")
            
            with c2:
                if st.button("🚀 開始分析", key="ana_btn_main"):
                    # 呼叫預測比對邏輯
                    process_analysis(selected_stock, predictions_ws)
            
            with c3:
                if st.button("🗑️ 刪除", key="del_btn_main"):
                    delete_stock(user_name, selected_stock, watchlist_ws)

# --- 支援功能：刪除與分析 ---

def delete_stock(user, symbol, ws):
    """刪除邏輯：找到對應列並移除"""
    try:
        all_data = ws.get_all_values()
        for i, row in enumerate(all_data):
            if row[0] == user and row[1] == symbol:
                ws.delete_rows(i + 1)
                st.success(f"已移除 {symbol}")
                st.rerun()
                return
    except Exception as e:
        st.error(f"刪除失敗: {e}")

def process_analysis(symbol, pred_ws):
    """分析邏輯：比對 predictions 表，決定是否呼叫 AI"""
    st.info(f"正在連線大腦分析 {symbol}...")
    
    # 1. 取得所有預測記錄
    preds = pred_ws.get_all_values()
    # 假設 A 欄是股票代號
    exist = any(row[0] == symbol for row in preds)
    
    if exist:
        st.success(f"✨ 找到 {symbol} 的現有記錄，正在讀取分析報告...")
        # 這裡後續可以串接讀取該列的數據
    else:
        st.warning(f"🧠 大腦資料庫無紀錄，正在為 {symbol} 新增 AI 分析任務...")
        # 這裡模擬 AI 寫入新資料
        pred_ws.append_row([symbol, "AI分析中...", "N/A"])
        st.success(f"🚀 任務已派發，請稍後查看。")


# 確保程式啟動
if __name__ == "__main__":
    main()










