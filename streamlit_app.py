import extra_streamlit_components as st_tags
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import json
import re
import requests # <-- 記得補上這行，因為發送指令需要它
import time     # <-- 記得補上這行，後續等待檢查需要它
# ==========================================
# 基礎設定章節：強制白色主題與解鎖
# ==========================================
def setup_page():
    st.set_page_config(page_title="智慧AI輔助", layout="centered")
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
# GitHub 連線通訊章節：叫醒雲端大腦
# ==========================================
def trigger_github_analysis(symbol):
    """發送 API 請求給 GitHub，啟動指定的 Workflow 並傳入股票代號"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        workflow = st.secrets["GITHUB_WORKFLOW_ID"]
        
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {
            "ref": "main", 
            "inputs": {"target_symbol": symbol}
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 204
    except Exception as e:
        st.error(f"連線 GitHub 失敗: {e}")
        return False
        
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
            all_users = db_ws.col_values(1)
            if u in all_users:
                st.error(f"❌ 帳號 '{u}' 已被使用")
            else:
                # 【關鍵修改】在帳號與密碼前加上單引號，保留開頭的 0
                db_ws.append_row([str(u), str(p)]) # 直接存，不加單引號
                st.success("🎉 註冊成功！請切換到登入分頁。")
        else:
            st.warning("請檢查輸入內容是否完整且格式正確。")

# ==========================================
# 第二章：帳號登入功能 (已整合寫入 Cookie)
# ==========================================
def chapter_2_login(db_ws, cookie_manager): # <-- 這裡多接收了參數
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
            # 2.4 核對邏輯
            data = db_ws.get_all_values()
            match = any(
                str(row[0]).strip().lstrip("'") == str(u).strip() and 
                str(row[1]).strip().lstrip("'") == str(p).strip() 
                for row in data
            )
            
            if match:
                # A. 原有的 Session 登入
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                
                # B. 【新增】寫入 Cookie 到瀏覽器，設定 14 天有效期
                import datetime
                expire_at = datetime.datetime.now() + datetime.timedelta(days=14)
                cookie_manager.set('oracle_remember_me', u, expires_at=expire_at)
                
                st.success("登入成功！正在跳轉...")
                st.rerun()
            else:
                st.error("❌ 帳號或密碼錯誤")

# ==========================================
# 核心執行入口章節 (終極修復 F5 登出問題)
# ==========================================
def main():
    setup_page()
    
    # 1. 初始化 Cookie 管理器
    cookie_manager = st_tags.CookieManager()
    
    # --- #2. 嘗試抓取瀏覽器記憶 (加入緩衝等待機制) ---
    saved_user = None
    
    # 初始化登出狀態標記
    if "just_logged_out" not in st.session_state:
        st.session_state["just_logged_out"] = False

    # 如果不是剛按過登出，就進入「循環讀取」邏輯
    if not st.session_state["just_logged_out"]:
        # 這裡的迴圈是為了解決 F5 重整時 Cookie 讀取過慢的問題
        # 我們最多等 1.2 秒 (0.3秒 * 4次)
        attempt = 0
        while saved_user is None and attempt < 4:
            saved_user = cookie_manager.get('oracle_remember_me')
            if saved_user:
                break
            import time
            time.sleep(0.3)
            attempt += 1
    # -----------------------------------------------

    # --- #3. 持久化判斷邏輯 (非阻塞優化版) ---
    if "logged_in" not in st.session_state:
        if saved_user:
            st.session_state["logged_in"] = True
            st.session_state["user"] = saved_user
            st.rerun()
        else:
            st.session_state["logged_in"] = False

    db_dict = init_db() 
    if db_dict is None: return

    # --- #4. 頁面顯示邏輯 ---
    if not st.session_state["logged_in"]:
        # 顯示歡迎標題
        st.markdown("<h1 style='text-align: center;'>🔮 股市輔助決策系統-進化型AI</h1>", unsafe_allow_html=True)
        
        # 【修正點】如果不是剛登出且沒抓到 Cookie，僅顯示小提醒而不卡死畫面
        if not st.session_state.get("just_logged_out", False) and saved_user is None:
            st.caption("ℹ️ 正在嘗試自動恢復連線... 若未跳轉請手動登入。")

        tab1, tab2 = st.tabs(["帳號登入", "帳號申請"])
        with tab1:
            chapter_2_login(db_dict["users"], cookie_manager)
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
                # 1. 強制刪除 Cookie (確保 Key 名稱完全一致)
                try:
                    cookie_manager.delete('oracle_remember_me')
                except:
                    pass
                
                # 2. 清除所有相關的 Session 狀態
                st.session_state["logged_in"] = False
                st.session_state["user"] = None
                
                # 3. 【關鍵】標記為剛登出，並立刻停止後續執行
                st.session_state["just_logged_out"] = True
                
                # 4. 給瀏覽器一點時間處理刪除指令後再重整
                import time
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")

        # 1. 執行第三章 (控制台與監控清單管理)
        chapter_3_watchlist_management(
            db_dict["users"], 
            db_dict["watchlist"], 
            db_dict["predictions"]
        )

        # 2. 獲取目前選中的股票 (從第三章的 radio 按鈕取得)
        selected_stock = st.session_state.get("stock_selector")
        
        if selected_stock:
            # 【核心修正】如果使用者在清單換了股票，但目前的報告還是舊股票的，就先清掉它
            # 這樣可以強迫使用者按下「開始分析」，進而觸發控制台的自動收合
            if "current_analysis" in st.session_state:
                if st.session_state["current_analysis"][1] != selected_stock:
                    st.session_state.pop("current_analysis")
            
            # 3. 執行第四章 (顯示即時行情觀測)
            chapter_4_stock_basic_info(selected_stock)

            # 4. 執行第五章 (AI 深度報告)
            # 只有當使用者點擊「開始分析」並成功取得結果 (存入 session_state) 後才會顯示
            if "current_analysis" in st.session_state:
                chapter_5_ai_decision_report(st.session_state["current_analysis"], db_dict["predictions"])
                    
# ==========================================
# 第三章：監控清單管理功能 (Control Panel) - 穩定收合版
# ==========================================
def chapter_3_watchlist_management(db_ws, watchlist_ws, predictions_ws):
    import yfinance as yf
    import datetime
    user_name = st.session_state["user"]
    
    # --- 防困邏輯 1：初始化展開狀態 ---
    if "menu_expanded" not in st.session_state:
        st.session_state["menu_expanded"] = True 

    # 1. 取得目前使用者的自選清單
    try:
        all_watch = watchlist_ws.get_all_values()
        user_stocks = [row[1] for row in all_watch if len(row) > 1 and row[0] == user_name]
    except Exception:
        user_stocks = []
    
    stock_count = len(user_stocks)

    # --- 3.1 關鍵：使用 session_state 直接驅動 expander ---
    with st.expander(f"🛠️ 股票控制台 ({stock_count}/20)", expanded=st.session_state["menu_expanded"]):
        
        # 3.2 上半部：新增功能
        st.write("### 📥 新增自選股")
        
        col_input, col_add = st.columns([3, 1])
        with col_input:
            new_stock = st.text_input("輸入股票代號 (英數)", key="new_stock_input").strip().upper()
        
        with col_add:
            st.write("##") # 對齊
            if st.button("確認新增", key="add_stock_btn"):
                # 新增前確保狀態設為 True，防止誤收合
                st.session_state["menu_expanded"] = True
                
                if not new_stock:
                    st.warning("⚠️ 請先輸入代號")
                elif not is_valid_format(new_stock):
                    st.error("🚫 格式錯誤：僅限輸入英文或數字")
                elif stock_count >= 20:
                    st.error("❌ 已達上限：最多只能 20 筆自選股。請先刪除不用的股票。")
                elif any(s.startswith(new_stock) for s in user_stocks):
                    st.info("💡 提醒：此股票已在清單中")
                else:
                    with st.spinner(f"🔍 正在驗證市場代號 {new_stock}..."):
                        suffix = ".TW" if len(new_stock) == 4 and new_stock[0] in ['2', '3'] else ".TWO"
                        full_code = f"{new_stock}{suffix}"
                        test_data = yf.Ticker(full_code).history(period="1d")
                        
                        if not test_data.empty:
                            watchlist_ws.append_row([user_name, full_code])
                            st.success(f"✅ {full_code} 已加入清單")
                            st.rerun()
                        else:
                            st.error(f"❌ 查無此股票代號 {new_stock}")

        st.markdown("---")
        
        # 3.4 下半部：清單管理
        st.write("### 📋 自選股清單")
        if not user_stocks:
            st.info("目前清單中沒有股票")
        else:
            selected_stock = st.radio(
                "選擇要操作的股票", 
                options=user_stocks, 
                key="stock_selector",
                horizontal=True
            )
            
            c2, c3 = st.columns(2)
            with c2:
                # 【開始分析按鈕】
                if st.button("🚀 開始分析", key="ana_btn_main", use_container_width=True):
                    # 第一步：立刻變更狀態為 False
                    st.session_state["menu_expanded"] = False
                    
                    with st.spinner("正在處理請求..."):
                        result = process_analysis(selected_stock, predictions_ws)
                        if result:
                            st.session_state["current_analysis"] = result
                    
                    # 第二步：帶領新的 False 狀態重整頁面，Expander 就會收起
                    st.rerun()
            
            with c3:
                # 【刪除按鈕】
                if st.button("🗑️ 刪除", key="del_btn_main", use_container_width=True):
                    # 刪除時確保狀態為 True，維持展開
                    st.session_state["menu_expanded"] = True
                    delete_stock(user_name, selected_stock, watchlist_ws)
                    # delete_stock 內部若有 rerun，會讀到上面的 True

# ==========================================
# 拼圖 A：顯示器 (專門解決你看到的紅字問題)
# ==========================================
def display_analysis_results(row):
    """將試算表數據轉化為漂亮圖表，若 row 不存在則不執行"""
    if not row or len(row) < 3:
        return
    
    st.markdown("---")
    st.success(f"### 🎯 AI 分析報告：{row[1]}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("分析日期", row[0])
    with col2:
        advice = row[2]
        color = "green" if "買" in advice else "red" if "賣" in advice else "orange"
        st.markdown(f"**建議方向：** :{color}[{advice}]")
    with col3:
        st.metric("AI 信心度", row[3] if len(row) > 3 else "N/A")

    with st.expander("📊 查看詳細 AI 運算指標 (共 37 項)"):
        # 顯示從第 5 欄開始的所有詳細技術指標
        st.write(row[4:])

def process_analysis(symbol, pred_ws):
    """
    全表定錨最終版：
    1. 保護期內 (23:00-14:30)：定錨於全表最新日期，若完全無紀錄則判定為今日新股。
    2. 非保護期 (14:31-22:59)：正常觸發 AI 大腦更新。
    """
    import datetime
    import time
    now = datetime.datetime.now()
    current_time = now.time()
    
    # 判斷是否處於「保護期」 (23:00 到 隔天 14:30)
    is_readonly_period = (current_time >= datetime.time(23, 0)) or (current_time <= datetime.time(14, 30))
    today_str = now.strftime("%Y-%m-%d")

    # 1. 抓取所有資料
    all_data = pred_ws.get_all_values()
    if not all_data or len(all_data) < 2:
        st.warning("試算表尚無任何數據。")
        return None

    # 2. 找出全表「最新的一個日期」 (作為保護期的定錨點)
    all_dates = [row[0] for row in all_data[1:] if row[0]]
    latest_date_in_sheet = max(all_dates) if all_dates else today_str

    # 3. 執行分支策略
    if is_readonly_period:
        # --- [保護期：讀取模式] ---
        # 搜尋 符合該股票 且 日期等於「全表最新日期」的那一行
        found_row = next((r for r in all_data if len(r) > 1 and str(r[1]).strip() == str(symbol).strip() and r[0] == latest_date_in_sheet), None)
        
        if found_row:
            st.success(f"📌 已載入定錨預測報告 (參考最新結算日: {latest_date_in_sheet})")
            return found_row
        else:
            # 如果找不到該日期的資料，檢查這支股票是否「完全沒歷史紀錄」
            all_history = [r for r in all_data if len(r) > 1 and str(r[1]).strip() == str(symbol).strip()]
            
            if not all_history:
                # 這是使用者昨天或剛加入的股票
                st.info(f"🆕 偵測到新加入股票：{symbol}")
                st.warning(f"⚠️ 試算表內尚無 {symbol} 的歷史定錨數據。")
                st.info("💡 因目前為保護期，請待今日 14:30 收盤後，再執行分析以建立首份報告。")
            else:
                # 雖然最新日期沒資料，但以前有算過，就拿最近的一次出來
                st.info(f"ℹ️ {latest_date_in_sheet} 無紀錄，載入該股最近一次歷史報告 ({all_history[-1][0]})")
                return all_history[-1]
            return None
            
    else:
        # --- [分析期：更新模式] ---
        # 看看今天是不是已經分析過了
        today_row = next((r for r in all_data if len(r) > 1 and str(r[1]).strip() == str(symbol).strip() and r[0] == today_str), None)
        
        if today_row:
            return today_row
        
        # 今天還沒算，通知 AI 大腦啟動
        if trigger_github_analysis(symbol):
            placeholder = st.empty()
            placeholder.info(f"⏳ 雲端大腦正在進行今日盤後運算 {symbol}...")
            for i in range(30):
                time.sleep(4)
                current_data = pred_ws.get_all_values()
                new_row = next((r for r in current_data if len(r) > 1 and str(r[1]).strip() == str(symbol).strip() and r[0] == today_str), None)
                if new_row:
                    placeholder.empty()
                    return new_row 
                placeholder.info(f"⏳ 雲端計算中... (進度: {i+1}/30)")
            placeholder.error("❌ 分析逾時，請檢查 GitHub Action 狀態")
        return None
                

# ==========================================
# 補強工人 1：格式檢查 (防止新增報錯)
# ==========================================
def is_valid_format(text):
    import re
    return bool(re.match("^[a-zA-Z0-9]*$", text))

# ==========================================
# 補強工人 2：刪除邏輯 (防止刪除報錯)
# ==========================================
def delete_stock(user, symbol, watchlist_ws):
    try:
        all_data = watchlist_ws.get_all_values()
        # 過濾掉該使用者要刪除的那支股票
        updated_rows = [all_data[0]] + [row for row in all_data[1:] if not (row[0] == user and row[1] == symbol)]
        
        watchlist_ws.clear()
        watchlist_ws.update('A1', updated_rows)
        st.success(f"🗑️ 已移除 {symbol}")
        import time
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"刪除失敗: {e}")

# ==========================================
# 第四章：基本行情觀測面板 (行情觀測站)
# ==========================================
def chapter_4_stock_basic_info(symbol):
    """
    獨立章節：顯示股票即時行情，具備手動更新機制以節省資源。
    """
    import yfinance as yf
 
    # 佈局：標題與更新按鈕
    col_info, col_refresh = st.columns([5, 1])
    with col_info:
        st.write(f"目前觀測對象：**{symbol}**")
    with col_refresh:
        # 手動更新按鈕：只有按下才觸發 yfinance 請求
        refresh_pushed = st.button("🔄 更新行情", key=f"refresh_ch4_{symbol}")

    # 使用 session_state 儲存數據，避免重複抓取被鎖 IP
    cache_key = f"ch4_data_{symbol}"
    
    if refresh_pushed or cache_key not in st.session_state:
        with st.spinner(f"正在連線市場獲取 {symbol} 最新報價..."):
            try:
                ticker = yf.Ticker(symbol)
                # 抓取 2 日數據以計算昨日與今日的變動
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    # 提取數據
                    prev_close = hist['Close'].iloc[-2]
                    open_price = hist['Open'].iloc[-1]
                    curr_price = hist['Close'].iloc[-1]
                    high_price = hist['High'].iloc[-1]
                    low_price = hist['Low'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    
                    change = curr_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # 寫入快取
                    st.session_state[cache_key] = {
                        "prev_close": prev_close,
                        "open_price": open_price,
                        "curr_price": curr_price,
                        "change": change,
                        "change_pct": change_pct,
                        "volume": volume,
                        "high": high_price,
                        "low": low_price
                    }
                else:
                    st.warning("⚠️ 查無足夠的交易數據（可能今日尚未開盤或停牌）")
                    return
            except Exception as e:
                st.error(f"行情抓取失敗：{e}")
                return

    # 從快取中顯示數據
    data = st.session_state.get(cache_key)
    if data:
        # 漲紅跌綠邏輯
        color = "red" if data["change"] >= 0 else "green"
        sign = "+" if data["change"] >= 0 else ""

        # --- 第一排資訊 ---
        c1, c2, c3 = st.columns(3)
        c1.write(f"昨日收盤：**{data['prev_close']:.2f}**")
        c2.write(f"今日開盤：**{data['open_price']:.2f}**")
        c3.write(f"當前價格：**:{color}[{data['curr_price']:.2f}]**")

        # --- 第二排資訊 ---
        c4, c5, c6 = st.columns(3)
        c4.write(f"漲跌價格：**:{color}[{sign}{data['change']:.2f}]**")
        c5.write(f"漲跌幅度：**:{color}[{sign}{data['change_pct']:.2f}%]**")
        c6.write(f"今日成交量：**{int(data['volume']):,}**")

    st.markdown("---") # 章節結束線

# ==========================================
# 第五章：AI 深度決策報告 (修正索引與防錯)
# ==========================================
def chapter_5_ai_decision_report(row, pred_ws):
    # --- 內部工具函數：放在這裡確保不會發生 NameError ---
    def safe_float(value):
        try:
            if value is None: return 0.0
            # 移除可能干擾轉換的符號
            clean_val = str(value).replace('%', '').replace(',', '').strip()
            if clean_val == "" or clean_val == "-": return 0.0
            return float(clean_val)
        except (ValueError, TypeError):
            return 0.0

    # --- 1. 標題與市場情緒 (抓取 AK 欄位索引 36) ---
    analysis_date = row[0]
    # 根據截圖 AK 欄位是索引 36
    sentiment_raw = row[36] if len(row) > 36 else "數據累積中"
    s_icon = "🧘" if "冷靜" in sentiment_raw else "🔥" if "過熱" in sentiment_raw else "📊"
    
    st.markdown(f"### 🔮 價格預演 (基準日：{analysis_date}) {s_icon} <small>{sentiment_raw}</small>", unsafe_allow_html=True)

    # --- 2. 核心預測數據 ---
    c1, c2 = st.columns(2)
    with c1:
        st.metric("預計收盤價", f"{row[2]}") 
        st.markdown(f"<p style='color:gray; font-size:0.9rem; margin-top:-15px;'>波動區間：{row[3]} ~ {row[4]}</p>", unsafe_allow_html=True)
    with c2:
        st.write("**AI 辨識信心度**")
        st.progress(0.9) # 這裡可改為動態比例
        st.caption("信心值：90.0%")

    st.markdown("---")

    # --- 2.5 策略預估價位表格 (補回此區塊) ---
    st.write("### 🎯 策略預估價位矩陣")
    
    # 根據試算表索引精確對應：
    # 建議買價：buy_5d(6), buy_10d(7), buy_20d(9)
    # 建議賣價：sell_5d(12), sell_10d(13), sell_20d(15)
    # 壓力價位：res_5d(18), res_10d(19), res_20d(21)
    # 乖離率：bias_5d(29), bias_10d(30), bias_20d(32)
    
    price_matrix = {
        "時序": ["5日建議", "10日建議", "20日建議"],
        "建議買價": [row[6], row[7], row[9]], 
        "建議賣價": [row[12], row[13], row[15]],
        "壓力價位": [row[18], row[19], row[21]],
        "乖離率 (%)": [
            f"{row[29]}%" if len(row) > 29 else "-",
            f"{row[30]}%" if len(row) > 30 else "-",
            f"{row[32]}%" if len(row) > 32 else "-"
        ]
    }
    
    # 使用 dataframe 顯示並隱藏索引，讓介面更專業
    st.dataframe(price_matrix, hide_index=True, use_container_width=True)
    
    st.markdown("---")

    # --- 3. 最新 10 筆預測準確率驗證 ---
    st.write("### 📈 最新 10 筆預測準確率驗證")
    try:
        all_data = pred_ws.get_all_values()
        symbol = row[1]
        history_rows = [r for r in all_data[1:] if len(r) > 1 and r[1] == symbol]
        display_rows = list(reversed(history_rows))[:10]
        
        if display_rows:
            acc_data = []
            for h_row in display_rows:
                # 實際收盤價在 Y 欄 (索引 24)
                h_actual = h_row[24] if (len(h_row) > 24 and h_row[24] not in ["", "0", "0.0"]) else "累積中..."
                # 準確率在 Z 欄 (索引 25)
                acc = "累積中..."
                if h_actual != "累積中...":
                    try:
                        err = safe_float(h_row[25])
                        acc = f"{100 - abs(err):.2f}%"
                    except: pass
                
                acc_data.append({
                    "預測日期": h_row[0],
                    "預測價格": h_row[2],
                    "實際收盤價": h_actual,
                    "準確率": acc
                })
            st.dataframe(acc_data, hide_index=True, use_container_width=True)
        else:
            st.info("💡 尚未有歷史預測數據")
    except Exception as e:
        st.caption(f"準確率加載中...")

    st.markdown("---")
    
    # --- 4. 核心指標儀表板 (精確索引對應 AH:33, AI:34, AJ:35) ---
    st.write("### 📊 核心戰略指標 (Oracle Strategy Metrics)")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        # AH 欄 (索引 33): atr_value
        atr_v = safe_float(row[33]) if len(row) > 33 else 0.0
        st.metric("股價活潑度 (ATR)", f"{atr_v:.2f}")
        st.caption("💡 數字大代表股價跳動大，機會多但洗盤也兇。")

    with col_b:
        # AI 欄 (索引 34): vol_bias
        vol_b = safe_float(row[34]) if len(row) > 34 else 0.0
        v_status = "🔥 資金湧入" if vol_b > 0 else "❄️ 動能不足"
        st.metric("資金追價意願", v_status, delta=f"{vol_b}%")
        st.caption("💡 正數代表大家肯拿錢追高；負數代表只是虛漲。")

    with col_c:
        # AJ 欄 (索引 35): rr_ratio
        rr_v = safe_float(row[35]) if len(row) > 35 else 0.0
        rr_txt = "💎 極具價值" if rr_v > 1.5 else "⚠️ 風險偏高"
        st.metric("投資性價比 (R/R)", rr_txt)
        st.caption(f"💡 目前為 {rr_v:.1f}。代表賠 1 塊的風險能換 {rr_v:.1f} 塊獲利。")

    st.markdown("---")

    # --- 5. AI 診斷與展望 (AB:27, AC:28) ---
    st.write("### 🧠 Oracle 深度診斷")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info(f"**【AI 臨床診斷】**\n\n{row[27]}")
    with col_d2:
        st.success(f"**【未來展望評估】**\n\n{row[28]}")

# 確保程式啟動
if __name__ == "__main__":
    main()








