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
        # 2. 【關鍵補位】執行第四章 (基本行情觀測)
        # 我們從 session_state 抓取使用者在第三章選中的股票
        selected_stock = st.session_state.get("stock_selector")
        if selected_stock:
            chapter_4_stock_basic_info(selected_stock)

        # 3. 執行第五章 (AI 深度報告)
            # 只有當我們有點擊「開始分析」取得結果後才顯示
            if "current_analysis" in st.session_state:
                # 確保分析的股票跟目前選中的股票是同一支
                if st.session_state["current_analysis"][1] == selected_stock:
                    chapter_5_ai_decision_report(st.session_state["current_analysis"], db_dict["predictions"])
                    
# ==========================================
# 第三章：監控清單管理功能 (Control Panel)
# ==========================================

def chapter_3_watchlist_management(db_ws, watchlist_ws, predictions_ws):
    import yfinance as yf
    import datetime
    user_name = st.session_state["user"]
    
    # --- 防困邏輯 1：初始化展開狀態 (僅在不存在時設定) ---
    if "menu_expanded" not in st.session_state:
        st.session_state["menu_expanded"] = True # 初始進入預設開啟

    # 1. 取得目前使用者的自選清單
    try:
        all_watch = watchlist_ws.get_all_values()
        user_stocks = [row[1] for row in all_watch if len(row) > 1 and row[0] == user_name]
    except Exception:
        user_stocks = []
    
    stock_count = len(user_stocks)

    # --- 3.1 使用變數控制 expanded 狀態 ---
    with st.expander("🛠️ 開啟股票控制台", expanded=st.session_state["menu_expanded"]):
        
        # 3.2 上半部：新增功能
        st.write(f"### 📥 新增自選股 ({stock_count}/20)")
        
        col_input, col_add = st.columns([3, 1])
        with col_input:
            new_stock = st.text_input("輸入股票代號 (英數)", key="new_stock_input").strip().upper()
        
        with col_add:
            st.write("##") # 對齊
            add_btn = st.button("確認新增", key="add_stock_btn")
            
        # 3.3 新增邏輯：維持展開狀態
        if add_btn:
            if not new_stock:
                st.warning("⚠️ 請先輸入代號")
            elif not is_valid_format(new_stock):
                st.error("🚫 格式錯誤：僅限輸入英文或數字")
            elif stock_count >= 20:
                st.warning("⚠️ 已達上限：最多只能 20 筆自選股")
            elif any(s.startswith(new_stock) for s in user_stocks):
                st.info("💡 提醒：此股票已在清單中")
            else:
                with st.spinner(f"🔍 正在驗證市場代號 {new_stock}..."):
                    # 簡易判斷台灣市場後綴
                    suffix = ".TW" if len(new_stock) == 4 and new_stock[0] in ['2', '3'] else ".TWO"
                    full_code = f"{new_stock}{suffix}"
                    
                    test_ticker = yf.Ticker(full_code)
                    test_data = test_ticker.history(period="1d")
                    
                    if not test_data.empty:
                        watchlist_ws.append_row([user_name, full_code])
                        st.success(f"✅ {full_code} 已加入清單")
                        # 防困：此處 rerun 會依據 session_state["menu_expanded"] (此時為 True) 保持開啟
                        st.rerun()
                    else:
                        st.error(f"❌ 查無此股票：市場中找不到代號 {new_stock}")

        st.markdown("---")
        
        # 3.4 下半部：清單管理
        st.write("### 📋 監控清單管理")
        if not user_stocks:
            st.info("目前清單中沒有股票")
        else:
            c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
            
            with c1:
                selected_stock = st.selectbox("選擇要操作的股票", options=user_stocks, key="stock_selector")
            
            with c2:
                if st.button("🚀 開始分析", key="ana_btn_main"):
                    with st.spinner("正在啟動 AI 運算..."):
                        result = process_analysis(selected_stock, predictions_ws)
                        if result:
                            st.session_state["current_analysis"] = result
                            # --- 關鍵防困：只有分析完成才將展開狀態設為 False ---
                            st.session_state["menu_expanded"] = False
                            st.rerun() 
            
            with c3:
                if st.button("🗑️ 刪除", key="del_btn_main"):
                    # 執行刪除，狀態維持為 True
                    delete_stock(user_name, selected_stock, watchlist_ws)

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

# ==========================================
# 拼圖 B：執行員 (完整覆蓋你給我的那段)
# ==========================================
def process_analysis(symbol, pred_ws):
    """
    靜默版執行員：負責背景同步與拿取數據，不直接顯示 UI
    """
    import time
    import yfinance as yf
    import datetime

    # 1. 取得市場最新收盤日
    try:
        stock_data = yf.Ticker(symbol)
        latest_market_date = stock_data.history(period="1d").index[0].strftime("%Y-%m-%d")
    except:
        latest_market_date = datetime.date.today().strftime("%Y-%m-%d")

    # 2. 搜尋表格
    all_data = pred_ws.get_all_values()
    found_row = next((row for row in all_data if len(row) > 1 and row[1] == symbol and row[0] == latest_market_date), None)

    if found_row:
        return found_row 
    else:
        # 3. 如果沒資料，安靜地觸發 GitHub
        if trigger_github_analysis(symbol):
            placeholder = st.empty() # 建立一個臨時顯示區
            placeholder.info(f"⏳ 雲端大腦正在計算 {symbol}，請稍候...")
            
            max_retries = 30
            for i in range(max_retries):
                time.sleep(4)
                current_data = pred_ws.get_all_values()
                new_row = next((r for r in current_data if len(r) > 1 and r[1] == symbol and r[0] == latest_market_date), None)
                
                if new_row:
                    placeholder.empty() # 成功後清除提示
                    return new_row 
                
                # 更新進度提示，確保縮排正確
                placeholder.info(f"⏳ 雲端計算中... (進度: {i+1}/{max_retries})")
            
            placeholder.error("❌ 分析逾時，請稍後再試")
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
# 第五章：AI 深度決策報告 (精簡專業版)
# ==========================================
def chapter_5_ai_decision_report(row, pred_ws):
    """
    row: 當前選定股票的預測數據
    pred_ws: predictions 分頁，用於抓取歷史準確率
    """
    if not row or len(row) < 33:
        st.error("數據欄位不足，請檢查試算表格式")
        return

    # --- 1. 標題區 (整合基準日，取代黃色大區塊) ---
    analysis_date = row[0]
    st.markdown(f"### 🔮 隔日價格預演 (分析基準日：{analysis_date})")

    # --- 2. 核心預測數據 ---
    c1, c2 = st.columns(2)
    with c1:
        # 顯示預計收盤價與區間 (同格上下行)
        st.metric("預計收盤價", f"{row[2]}") 
        st.markdown(f"<p style='color:gray; font-size:0.9rem; margin-top:-15px;'>波動區間：{row[3]} ~ {row[4]}</p>", unsafe_allow_html=True)
    with c2:
        # AI 信心度 (預設從試算表抓取或設定)
        st.write("**AI 辨識信心度**")
        conf_val = 90.0 
        st.progress(conf_val / 100)
        st.caption(f"目前模型運算信心值為 {conf_val}%")

    st.markdown("---")

    # --- 3. 策略預估價位矩陣 (5/10/20日) ---
    st.write("### 🎯 策略預估價位矩陣")
    price_matrix = {
        "時序": ["5日建議", "10日建議", "20日建議"],
        "建議買價": [row[6], row[7], row[9]], 
        "建議賣價": [row[12], row[13], row[15]],
        "壓力價位": [row[18], row[19], row[21]],
        "乖離率 (%)": [row[29], row[30], row[32]]
    }
    st.table(price_matrix)

    # --- 4. 歷史準確率驗證 (隱藏索引 0，僅限 10 筆) ---
    st.write("### 📈 最新 10 筆預測準確率驗證")
    try:
        all_data = pred_ws.get_all_values()
        symbol = row[1]
        history_rows = [r for r in all_data[1:] if len(r) > 1 and r[1] == symbol]
        display_rows = list(reversed(history_rows))[:10]
        
        if display_rows:
            accuracy_data = []
            for h_row in display_rows:
                h_actual = h_row[24] if (len(h_row) > 24 and h_row[24] not in ["", "0", "0.0", None]) else "累積中..."
                acc = "累積中..."
                if h_actual != "累積中...":
                    try:
                        err = float(h_row[25])
                        acc = f"{100 - abs(err):.2f}%"
                    except:
                        pass
                
                accuracy_data.append({
                    "預測日期": h_row[0],
                    "預測價格": h_row[2],
                    "實際收盤價": h_actual,
                    "準確率": acc
                })
            
            # 使用 dataframe 顯示並隱藏左側索引 0
            st.dataframe(accuracy_data, hide_index=True, use_container_width=True)
        else:
            st.info("💡 尚未有歷史預測數據")
            
    except Exception as e:
        st.caption(f"數據讀取中... ({e})")

    st.markdown("---")

    # --- 核心指標儀表板 ---
st.write("### 📊 核心戰略指標 (Oracle Strategy Metrics)")

# 根據截圖校正索引：AH[33], AI[34], AJ[35]
col_a, col_b, col_c = st.columns(3)

def safe_float(value):
    """安全轉換數值函數，避免非數字字元導致崩潰"""
    try:
        # 移除百分比符號或空格
        clean_val = str(value).replace('%', '').strip()
        return float(clean_val)
    except (ValueError, TypeError):
        return 0.0

with col_a:
    # AH 欄 (索引 33): atr_value
    atr_val = safe_float(row[33]) if len(row) > 33 else 0.0
    st.metric("股價活潑度 (ATR)", f"{atr_val:.2f}")
    st.caption("💡 數字越大代表股價跳動劇烈，獲利空間大但洗盤風險也高。")

with col_b:
    # AI 欄 (索引 34): vol_bias
    vol_b = safe_float(row[34]) if len(row) > 34 else 0.0
    status = "🔥 資金湧入" if vol_b > 0 else "❄️ 動能不足"
    st.metric("資金追價意願", status, delta=f"{vol_b}%")
    st.caption("💡 正數代表漲起來很有力；負數代表只是虛漲，追價意願低。")

with col_c:
    # AJ 欄 (索引 35): rr_ratio
    rr_val = safe_float(row[35]) if len(row) > 35 else 0.0
    # 專業風報比判斷
    if rr_val > 2.0:
        rr_status = "💎 極具價值"
    elif rr_val > 1.0:
        rr_status = "⚖️ 比例合理"
    else:
        rr_status = "⚠️ 風險偏高"
    
    st.metric("投資性價比 (R/R)", rr_status)
    st.caption(f"💡 目前為 {rr_val:.2f}。代表每承擔 1 份風險，預期換回 {rr_val:.2f} 份獲利。")

st.markdown("---")

# --- 5. AI 診斷與展望 (對應 AB[27], AC[28]) ---
st.write("### 🧠 Oracle 深度診斷")
col_diag, col_out = st.columns(2)
with col_diag:
    # AB 欄 (索引 27)
    st.info(f"**【AI 臨床診斷】**\n\n{row[27]}")
with col_out:
    # AC 欄 (索引 28)
    st.success(f"**【未來展望評估】**\n\n{row[28]}")


# 確保程式啟動
if __name__ == "__main__":
    main()
