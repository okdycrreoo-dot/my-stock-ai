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
    import yfinance as yf
    import datetime
    user_name = st.session_state["user"]
    
    # 1. 取得目前使用者的自選清單
    try:
        all_watch = watchlist_ws.get_all_values()
        # A 欄是 User, B 欄是股票代號
        user_stocks = [row[1] for row in all_watch if len(row) > 1 and row[0] == user_name]
    except:
        user_stocks = []
    
    stock_count = len(user_stocks)

    # --- 3.1 整個功能都裝進縮放按鈕 ---
    with st.expander("🛠️ 開啟股票控制台", expanded=False):
        
        # 3.2 上半部：新增功能佈局
        # [個人化指令實現]：上限設為 20，並顯示提醒
        st.write(f"### 📥 新增自選股 ({stock_count}/20)")
        
        col_input, col_add = st.columns([3, 1])
        
        with col_input:
            new_stock = st.text_input("輸入股票代號 (英數)", key="new_stock_input").strip().upper()
        
        with col_add:
            st.write("##") # 對齊
            add_btn = st.button("確認新增", key="add_stock_btn")
            
        # 3.3 新增邏輯處理 (強化比對與驗證)
        if add_btn:
            if not new_stock:
                st.warning("⚠️ 請先輸入代號")
            elif not is_valid_format(new_stock): # 保留你原本的格式檢查函數
                st.error("🚫 格式錯誤：僅限輸入英文或數字")
            elif stock_count >= 20:
                st.warning("⚠️ 已達上限：最多只能 20 筆自選股")
            elif any(s.startswith(new_stock) for s in user_stocks):
                st.info("💡 提醒：此股票已在清單中")
            else:
                # --- 新增：市場代號存在性校驗邏輯 ---
                with st.spinner(f"🔍 正在驗證市場代號 {new_stock}..."):
                    # 判斷邏輯：嘗試 .TW 或 .TWO，確保代號真實存在
                    if len(new_stock) == 4 and new_stock[0] in ['2', '3']:
                        suffix = ".TW"
                    else:
                        suffix = ".TWO"
                    
                    full_code = f"{new_stock}{suffix}"
                    
                    # 檢查 yfinance 是否能抓到歷史資料
                    test_ticker = yf.Ticker(full_code)
                    test_data = test_ticker.history(period="1d")
                    
                    if not test_data.empty:
                        # 只有真實存在的股票才會寫入
                        watchlist_ws.append_row([user_name, full_code])
                        st.success(f"✅ {full_code} 已加入清單")
                        st.rerun()
                    else:
                        st.error(f"❌ 查無此股票：市場中找不到代號 {new_stock}")

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
                    # 呼叫改進後的分析與喚醒邏輯
                    process_analysis(selected_stock, predictions_ws)
            
            with c3:
                if st.button("🗑️ 刪除", key="del_btn_main"):
                    delete_stock(user_name, selected_stock, watchlist_ws)

# --- 支援功能：刪除與分析 (完全覆蓋版) ---

def process_analysis(symbol, pred_ws):
    """
    ST 純偵查模式：
    1. 只判斷最新資料是否存在於 predictions 表。
    2. 若有：直接顯示。
    3. 若無：不寫入表格，顯示「分析中」，並由大腦外部觸發機制進行回填。
    """
    import time
    import yfinance as yf
    import datetime

    st.info(f"🔍 正在核對 {symbol} 數據狀態...")
    
    # 1. 取得市場最新收盤日
    try:
        stock_data = yf.Ticker(symbol)
        latest_market_date = stock_data.history(period="1d").index[0].strftime("%Y-%m-%d")
    except:
        latest_market_date = datetime.date.today().strftime("%Y-%m-%d")

    # 2. 搜尋 predictions (純讀取比對)
    all_data = pred_ws.get_all_values()
    found_row = None
    
    for row in all_data:
        if len(row) > 1 and row[1] == symbol and row[0] == latest_market_date:
            found_row = row
            break

    # 3. 執行判斷
    if found_row:
        st.success(f"✅ 取得最新分析報告 ({latest_market_date})")
        display_analysis_results(found_row)
        
    else:
        # --- 修正開始：使用 placeholder 解決字樣重複跳出問題 ---
        with st.status(f"🔮 偵測到 {symbol} 需更新，AI 大腦已接手...", expanded=True) as status:
            st.write("🧠 AI 正在進行深度分析並回填 37 項指標...")
            
            # 建立一個專門放「同步中」字樣的容器
            sync_text_area = st.empty()
            sync_text_area.write("⏳ 正在同步數據中...")
            
            # --- 4. 輪詢 (Polling)：持續檢查表格直到大腦寫入完畢 ---
            max_retries = 30 
            success = False
            for i in range(max_retries):
                time.sleep(3) 
                
                current_all_data = pred_ws.get_all_values()
                newly_written_row = next((r for r in current_all_data if len(r) > 1 and r[1] == symbol and r[0] == latest_market_date), None)
                
                if newly_written_row:
                    success = True
                    sync_text_area.empty() # 同步成功，清除字樣
                    status.update(label="✅ 大腦寫入完成！", state="complete", expanded=False)
                    st.success(f"✨ {symbol} 分析成功")
                    display_analysis_results(newly_written_row)
                    return
                
                # 這裡原本會一直 st.write 導致字串堆疊，現在透過 sync_text_area.write 保持只有一行
                if i % 3 == 0:
                    sync_text_area.write("⏳ 正在同步數據中...")
            
            if not success:
                sync_text_area.empty() # 失敗也清除該字樣
                status.update(label="❌ 分析逾時", state="error")
                st.error("❌ 同步失敗，請連絡管理者。")
# 確保程式啟動
if __name__ == "__main__":
    main()


