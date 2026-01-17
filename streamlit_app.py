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

import time
import yfinance as yf

# --- 支援功能：刪除與分析 ---

def delete_stock(user, symbol, ws):
    """刪除邏輯：找到對應列並移除"""
    try:
        all_data = ws.get_all_values()
        for i, row in enumerate(all_data):
            # A 欄是 User, B 欄是 Symbol
            if len(row) > 1 and row[0] == user and row[1] == symbol:
                ws.delete_rows(i + 1)
                st.success(f"已從自選清單移除 {symbol}")
                st.rerun()
                return
    except Exception as e:
        st.error(f"刪除失敗: {e}")

def process_analysis(symbol, pred_ws):
    """
    ST 背景邏輯確認：
    1. 比對 predictions 中股票是否存在且日期最新。
    2. 若是，直接顯示，不叫 AI。
    3. 若否，發出『喚醒訊號』請大腦處理，並在 ST 顯示分析中。
    4. 僅做判斷與喚醒，不重複寫入相同股票。
    """
    import time
    import yfinance as yf
    import datetime

    st.info(f"🔍 正在核對 {symbol} 的資料庫狀態...")
    
    # --- 1. 取得市場最新收盤日 ---
    try:
        stock_data = yf.Ticker(symbol)
        # 抓取最後一個交易日的日期
        latest_market_date = stock_data.history(period="1d").index[0].strftime("%Y-%m-%d")
    except:
        latest_market_date = datetime.date.today().strftime("%Y-%m-%d")

    # --- 2. 搜尋 predictions 內容 (背景邏輯確認) ---
    all_data = pred_ws.get_all_values()
    row_idx = -1
    is_latest = False
    
    for i, row in enumerate(all_data):
        # B 欄是代號 (index 1)
        if len(row) > 1 and row[1] == symbol:
            row_idx = i + 1 # 紀錄找到的行數
            # A 欄是收盤日期 (index 0)
            if row[0] == latest_market_date:
                is_latest = True
            break # 重要：找到第一筆就停止，避免重複

    # --- 3. 執行判斷與喚醒 ---
    if row_idx != -1 and is_latest:
        # 【情境一】資料已存在且日期最新：直接用現有的，不叫 AI
        st.success(f"✅ {symbol} 已有最新分析資料 ({latest_market_date})")
        display_analysis_results(all_data[row_idx-1])
        
    else:
        # 【情境二】資料不符或不存在：通知大腦出來處理
        with st.status("🔮 Oracle AI 正在分析中，請稍候...", expanded=True) as status:
            if row_idx != -1:
                # 股票存在但日期舊了：更新該行 F 欄標註，喚醒大腦
                pred_ws.update_cell(row_idx, 6, "Waiting Update")
                st.write(f"🔄 偵測到舊資料，已發送喚醒訊號請大腦更新...")
            else:
                # 完全沒資料：新增一列讓大腦填寫
                new_row = [""] * 37
                new_row[0] = latest_market_date # A: 日期
                new_row[1] = symbol             # B: 代號
                new_row[5] = "Waiting New"      # F: Status
                pred_ws.append_row(new_row)
                st.write(f"🆕 資料庫無紀錄，已請大腦直接處理新資料...")
                # 重新獲取最後一行的行號
                row_idx = len(pred_ws.get_all_values())

            # --- 4. 輪詢檢查 (Polling)：等待大腦補完 A-AK 欄位 ---
            # 這裡大腦會繞過 14:30 的限制，直接更新這行
            for _ in range(30): # 最多等 60 秒
                time.sleep(2) 
                updated_row = pred_ws.row_values(row_idx)
                
                # 檢查大腦是否寫完：F 欄狀態不再是 Waiting 且 A 欄日期正確
                if len(updated_row) >= 6 and updated_row[5] not in ["Waiting Update", "Waiting New", "AI分析中..."]:
                    status.update(label="✅ 大腦分析完成！", state="complete", expanded=False)
                    st.success(f"✨ {symbol} 資料已同步完成")
                    display_analysis_results(updated_row)
                    return
            
            status.update(label="❌ 分析逾時", state="error")
            st.error("大腦處理較久，請稍後刷新頁面查看。")

def display_analysis_results(data_row):
    """
    這裡負責將 A-AK 的 37 欄位資料視覺化
    """
    st.markdown("---")
    st.subheader(f"📊 {data_row[1]} 預測報告 ({data_row[0]})")
    
    # 這裡顯示 A-AK 完整資訊的排版
    # 舉例顯示前幾個欄位
    c1, c2, c3 = st.columns(3)
    c1.metric("收盤日期", data_row[0])
    c2.metric("預測收盤價", data_row[2] if data_row[2] else "--")
    c3.metric("狀態", data_row[5])
    
    # 暫時印出完整 row 確保開發者確認 37 欄位內容
    with st.expander("查看完整 37 欄原始數據 (A-AK)"):
        st.write(data_row)
# 確保程式啟動
if __name__ == "__main__":
    main()












