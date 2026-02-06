import extra_streamlit_components as st_tags
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import json
import re
import requests # <-- 記得補上這行，因為發送指令需要它
import time     # <-- 記得補上這行，後續等待檢查需要它
import google.generativeai as genai
# ==========================================
# 基礎設定章節：強制白色主題與解鎖
# ==========================================
def setup_page():
    st.set_page_config(page_title="AI智能自我進化中", layout="centered")
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
    return bool(re.match("^[a-zA-Z0-9]+$", text))
def safe_float(value):
    """安全轉換浮點數，處理空值、百分比符號與逗號"""
    try:
        if value is None: return 0.0
        # 如果是字串，先清理掉 % 和 ,
        if isinstance(value, str):
            value = value.replace('%', '').replace(',', '').strip()
        # 處理空字串
        if value == "": return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0
   
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

def trigger_admin_manual_sync():
    """【新增】管理者專用：啟動整個 YML 進行全量同步/修復"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        # 注意：全域觸發使用的是 dispatches 接口，不是 workflows/{id}/dispatches
        url = f"https://api.github.com/repos/{repo}/dispatches"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        # event_type 必須與 YML 中的 repository_dispatch -> types 一致
        data = {"event_type": "manual_trigger"}
        
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 204
    except Exception as e:
        st.error(f"管理員指令發送失敗: {e}")
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
            st.markdown(f"<h5 style='margin:0; white-space:nowrap;'>✅系統版本：20260202，歡迎回來，{st.session_state['user']}！</h5>", unsafe_allow_html=True)
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

        # --- 【核心進化：Oracle 算法雷達 - 完全覆蓋版 (排序與結構修正)】 ---
        predictions_ws = db_dict.get("predictions") 
        if predictions_ws is not None:
            try:
                import pandas as pd
                # 1. 抓取所有資料並轉換為 DataFrame
                raw_data = predictions_ws.get_all_values()
                if len(raw_data) > 1:
                    df_all = pd.DataFrame(raw_data[1:], columns=raw_data[0])
                    
                    # 【日期鎖定】：只取最後一個日期的資料，解決日期刷屏問題
                    latest_date = df_all.iloc[-1, 0] 
                    df_oracle = df_all[df_all.iloc[:, 0] == latest_date].copy()
                    
                    # 2. 定義內建判定函數
                    def check_strike_zone(row_series):
                        row = row_series.tolist()
                        
                        # Y 欄(24):實際價, AI 欄(34):資金, AD 欄(29):乖離, D 欄(3):支撐, S 欄(18):壓力
                        price = safe_float(row[24]) if len(row) > 24 else 0.0
                        if price == 0: price = safe_float(row[2]) 
                        
                        low_bound = safe_float(row[3]) if len(row) > 3 else 0.0   
                        bias_v = safe_float(row[29]) if len(row) > 29 else 0.0    
                        m_val = safe_float(row[34]) if len(row) > 34 else 0.0     
                        res_v = safe_float(row[18]) if len(row) > 18 else 9999.0  
                        
                        # 三層防護判定 (趨勢、資金、空間)
                        trend_ok = (price > low_bound) and (bias_v < 8)
                        money_ok = (m_val > 1.0)
                        space_ok = ((res_v - price) / price) > 0.03 if price > 0 else False
                        
                        return trend_ok and money_ok and space_ok

                    # 3. 執行邏輯掃描
                    strike_mask = df_oracle.apply(check_strike_zone, axis=1)
                    
                    # 【排序與去重】：抓取第 1 欄 (代號)，先 unique 再用 sorted 進行小到大排序
                    raw_list = df_oracle[strike_mask].iloc[:, 1].unique().tolist()
                    strike_list = sorted(raw_list) 
                    
                    if strike_list:
                        st.info(f"🎯 **Oracle 核心偵測 ({latest_date})：💎 絕佳擊球點！**\n\n`{'`, `'.join(strike_list)}`")
                    else:
                        st.caption(f"🔍 雷達掃描 ({latest_date})：目前尚未發現符合三位一體之目標。")
                else:
                    st.caption("🔍 雷達待命中：資料庫目前尚無預測資料。")

            except Exception as e:
                # 這是防止之前報錯的關鍵，必須要有 except 區塊
                # st.write(f"Radar Debug: {e}") 
                pass 
        # ----------------------------------------------------

        
        # 1. 執行第三章 (控制台與監控清單管理)
        chapter_3_watchlist_management(
            db_dict["users"], 
            db_dict["watchlist"], 
            db_dict["predictions"]
        )

        # --- 2. 核心修正：判斷顯示條件 ---
        # active_stock 是「按下分析按鈕後」鎖定的股票
        # current_selection 是「目前 Radio 選中」的股票
        active_stock = st.session_state.get("target_analysis_stock")
        current_selection = st.session_state.get("stock_selector")

        if active_stock:
            # 只有當「目前選的」跟「分析過的」是同一支，才顯示行情和報告
            if active_stock == current_selection:
                # 3. 執行第四章 (顯示即時行情觀測)
                chapter_4_stock_basic_info(active_stock)

                # 4. 執行第五章 (AI 深度報告)
                if "current_analysis" in st.session_state:
                    # 執行原有的 AI 報告 (內含第六章 Oracle 翻譯官)
                    chapter_5_ai_decision_report(st.session_state["current_analysis"], db_dict["predictions"])
                    # --- 【新增：第七章入口】 ---
                    # 這裡直接傳入 active_stock 和當前的分析行 row
                    chapter_7_ai_committee_analysis(active_stock, st.session_state["current_analysis"])
            else:
                # 如果使用者切換了 Radio 但還沒按分析按鈕
                st.info(f"💡 您切換到了 {current_selection}，請點擊「開始分析」以更新下方報表。")
                    
# ==========================================
# 第三章：監控清單管理功能 (Control Panel) - 邏輯修正版
# ==========================================
def chapter_3_watchlist_management(db_ws, watchlist_ws, predictions_ws):
    import yfinance as yf
    user_name = st.session_state["user"]
    
    # --- 防困邏輯 1：初始化展開狀態 ---
    if "menu_expanded" not in st.session_state:
        st.session_state["menu_expanded"] = True 

    # 1. 取得目前使用者的自選清單
    try:
        all_watch = watchlist_ws.get_all_values()
        user_stocks = [row[1] for row in all_watch if len(row) > 1 and row[0] == user_name]
        # --- 【新增：排序邏輯】 ---
        # 使用 sort() 會讓代號由小到大排列（例如：1101.TW -> 2330.TW -> 8046.TW）
        user_stocks.sort()
    except Exception:
        user_stocks = []
    stock_count = len(user_stocks)

    # --- 3.1 穩定且顯眼版控制台 (加強 CSS 權限) ---
    st.markdown("""
        <style>
        /* 1. 強制修改 expander 標題列背景與文字 */
        div[data-testid="stExpander"] details summary {
            background-color: #1E88E5 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        /* 2. 確保標題內的文字 P 標籤也是白色 */
        div[data-testid="stExpander"] details summary p {
            color: white !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
        }
        /* 3. 強制旋轉箭頭變白色 */
        div[data-testid="stExpander"] details summary svg {
            fill: white !important;
            color: white !important;
        }
        /* 4. 滑鼠移上去稍微變深藍 */
        div[data-testid="stExpander"] details summary:hover {
            background-color: #1565C0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    panel_label = f"🛠️ 股票控制台 (管理員模式)" if user_name == "admin" else f"🛠️ 股票控制台 ({stock_count}/20)"
    # 保持你現有的穩定緩衝邏輯
    current_expand_state = st.session_state.get("menu_expanded", True)
    with st.expander(panel_label, expanded=current_expand_state):
        
        # 3.2 上半部：新增功能
        st.write("### 📥 新增自選股")
        
        col_input, col_add = st.columns([3, 1])
        with col_input:
            new_stock = st.text_input("輸入股票代號 (英數)", key="new_stock_input").strip().upper()
        
        with col_add:
            st.write("##") 
            if st.button("確認新增", key="add_stock_btn"):
                st.session_state["menu_expanded"] = True
                if not new_stock:
                    st.warning("⚠️ 請先輸入代號")
                elif not is_valid_format(new_stock):
                    st.error("🚫 格式錯誤：僅限輸入英文或數字")
                # --- 權限分級：admin 無上限，一般使用者限制 20 支 ---
                elif user_name != "admin" and stock_count >= 20:
                    st.error("❌ 已達上限：一般帳戶最多只能 20 筆自選股。請先刪除不用的股票。")
                elif any(s.startswith(new_stock) for s in user_stocks):
                    st.info("💡 提醒：此股票已在清單中")
                else:
                    with st.spinner(f"🔍 正在跨市場驗證代號 {new_stock}..."):
                        # 1. 定義嘗試清單：先試上市(.TW)，再試上櫃(.TWO)
                        # 如果你有特殊代碼需求(如 ^TWII)，也可以把 new_stock 直接加進去
                        possible_codes = [f"{new_stock}.TW", f"{new_stock}.TWO"]
                        valid_full_code = None
                        
                        # 2. 開始循環嘗試 (輕量化穩定版：防止 IP 被封鎖)
                        for code in possible_codes:
                            try:
                                t = yf.Ticker(code)
                                # 優先檢查 fast_info，這不消耗 history 請求配額
                                if t.fast_info.get('last_price') is not None:
                                    valid_full_code = code
                                    break
                                # 若 fast_info 失敗，才試抓 1 天資料
                                test_data = t.history(period="1d")
                                if not test_data.empty:
                                    valid_full_code = code
                                    break
                            except:
                                continue
                        
                        # 3. 根據驗證結果執行寫入
                        if valid_full_code:
                            watchlist_ws.append_row([user_name, valid_full_code])
                            st.success(f"✅ {valid_full_code} 已加入清單")
                            st.rerun()
                        else:
                            st.error(f"❌ 驗證失敗：在上市(.TW)與上櫃(.TWO)皆查無代號 {new_stock}")

        st.markdown("---")
        
        # 3.4 下半部：清單管理
        st.write("### 📋 自選股清單")
        if not user_stocks:
            st.info("目前清單中沒有股票")
        else:
            # 此 radio 僅作選取，不直接觸發下方章節
            selected_in_radio = st.radio(
                "選擇要操作的股票", 
                options=user_stocks, 
                key="stock_selector",
                horizontal=True
            )
            
            # 將比例拉開到 4:1，讓刪除按鈕變得很窄
            c2, c3 = st.columns([4, 1])
            with c2:
                # 在文字前後加上各 3 個 \n，這會強制讓按鈕本體的「肉」變厚
                # 視覺上按鈕會比原本高出約 3-4 倍
                huge_btn_text = "🚀 開始分析 🚀 \n\n\n (點此執行)"
                
                if st.button(huge_btn_text, key="ana_btn_main", use_container_width=True, type="primary"):
                    st.session_state["target_analysis_stock"] = selected_in_radio
                    st.session_state["menu_expanded"] = False
                    
                    with st.spinner("正在進行深度分析..."):
                        result = process_analysis(selected_in_radio, predictions_ws)
                        if result:
                            st.session_state["current_analysis"] = result
                    st.rerun()

            with c3:
                # 刪除按鈕保持原樣，不加換行，它就會維持扁扁的
                if st.button("🗑️ 刪除", key=f"del_simple_{selected_in_radio}", use_container_width=True):
                    st.session_state["menu_expanded"] = True
                    delete_stock(user_name, selected_in_radio, watchlist_ws)
                    
        
        # === 3.5 管理者隱藏控制區 ===
        if st.session_state.get("user") == "admin":
            st.markdown("---")
            st.markdown("<p style='color:#FF4B4B; font-weight:bold;'>🔒 管理者專用後台</p>", unsafe_allow_html=True)
            col_adm, _ = st.columns([2, 1])
            with col_adm:
                if st.button("🔄 啟動 AI 全量補修 (Get Hub Action) ", key="admin_manual_trigger"):
                    with st.spinner("正在喚醒雲端大腦..."):
                        if trigger_admin_manual_sync():
                            st.success("✅ 指令已送出！GitHub 正在執行修補程序。")
                            st.toast("系統已接收指令，請稍後重整。")
                            time.sleep(2)
                            st.rerun() # 重置按鈕狀態
                        else:
                            st.error("❌ 觸發失敗。請檢查 Secrets 設定。")
                    
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
    """1.5 & 2.5 限制章節：僅限英數"""
    if not text: return False
    return bool(re.match("^[a-zA-Z0-9]+$", text))

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
                # 抓 7 天確保資料量足夠跨越假日
                hist = ticker.history(period="7d")
                
                if not hist.empty and len(hist) >= 2:
                    # 強制定位：最後一列絕對是「今天」，倒數第二列絕對是「昨天」
                    today_data = hist.iloc[-1]
                    yesterday_data = hist.iloc[-2]
                    
                    # 1. 昨收：直接取昨天的 Close
                    prev_close = yesterday_data['Close']
                    
                    # 2. 今開：直接取今天的 Open (對應你說的 10.60)
                    open_price = today_data['Open']
                    
                    # 3. 當前價：取今天的 Close (盤中會是即時價)
                    curr_price = today_data['Close']
                    
                    # 4. 成交量：股轉張
                    vol_in_lots = int(today_data['Volume'] / 1000)
                    
                    st.session_state[cache_key] = {
                        "prev_close": prev_close,
                        "open_price": open_price,
                        "curr_price": curr_price,
                        "change": curr_price - prev_close,
                        "change_pct": ((curr_price - prev_close) / prev_close * 100) if prev_close != 0 else 0,
                        "volume": vol_in_lots,
                        "high": today_data['High'],
                        "low": today_data['Low']
                    }
                else:
                    # 3. 終極救援：如果 history 完全沒資料，改用 fast_info 抓即時價
                    f_price = ticker.fast_info.get('last_price')
                    if f_price:
                        st.session_state[cache_key] = {
                            "prev_close": f_price, "open_price": f_price, "curr_price": f_price,
                            "change": 0, "change_pct": 0, "volume": 0, "high": f_price, "low": f_price
                        }
                    else:
                        st.warning(f"⚠️ 暫時無法獲取 {symbol} 的市場數據")
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
        c6.write(f"今日成交量：**{data['volume']:,} 張**")

    st.markdown("---") # 章節結束線

# ==========================================
# 第五章：AI 深度決策報告 (最終修正版)
# ==========================================
def chapter_5_ai_decision_report(row, pred_ws):
    # --- 內部工具函數 ---
    def safe_float(value):
        try:
            if value is None: return 0.0
            clean_val = str(value).replace('%', '').replace(',', '').strip()
            if clean_val == "" or clean_val == "-": return 0.0
            return float(clean_val)
        except (ValueError, TypeError):
            return 0.0

    # --- 1. 標題與市場情緒 (抓取 AK 欄位索引 36) ---
    analysis_date = row[0]
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
        raw_conf = row[37] if len(row) > 37 else ""
        
        if raw_conf in ["", "0", "0.0", None]:
            st.warning("⏳ 數據同步中...")
        else:
            conf_score = safe_float(raw_conf)
            display_conf = conf_score / 100 if conf_score > 1 else conf_score
            st.progress(min(max(display_conf, 0.0), 1.0)) 
            st.caption(f"信心值：{display_conf * 100:.1f}%")
    
    st.markdown("---")

    # --- 2.5 策略預估價位表格 ---
    st.write("### 🎯 策略預估價位矩陣")
    
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
    st.dataframe(price_matrix, hide_index=True, use_container_width=True)
    
    st.markdown("---")

    # --- 3. 最新 10 筆預測準確率驗證 (精準過濾版) ---
    st.write("### 📈 最新 10 筆預測準確率驗證")
    try:
        all_data = pred_ws.get_all_values()
        symbol = row[1]
        # 過濾該股票的歷史資料
        history_rows = [r for r in all_data[1:] if len(r) > 1 and r[1] == symbol]
        display_rows = list(reversed(history_rows))[:10]
        
        if display_rows:
            acc_data = []
            for h_row in display_rows:
                # 1. 抓取 F 欄 (索引 5) 的原始數值並清洗
                raw_val = str(h_row[5]).strip() if len(h_row) > 5 else ""
                
                # 2. 判斷是否為無效數據 (空值、0、或是包含"更新"、"累積"字眼)
                is_invalid = (
                    raw_val in ["", "0", "0.0", "None", "-"] or 
                    "更新" in raw_val or 
                    "累積" in raw_val
                )
                
                if is_invalid:
                    h_actual = "累積中..."
                    acc_display = "累積中..."
                else:
                    h_actual = raw_val
                    # 只有數據有效時，才計算準確率
                    try:
                        # 抓取 Z 欄 (索引 25) 的誤差百分比
                        err_val = h_row[25] if len(h_row) > 25 else "0"
                        err = safe_float(err_val)
                        
                        # 如果誤差和價格異常吻合但數值過小，也做防呆
                        if err == 0 and h_actual == "累積中...":
                            acc_display = "累積中..."
                        else:
                            acc_display = f"{100 - abs(err):.2f}%"
                    except:
                        acc_display = "累積中..."
                
                acc_data.append({
                    "預測日期": h_row[0],
                    "預測價格": h_row[2],
                    "實際收盤價": h_actual,
                    "準確率": acc_display
                })
            st.dataframe(acc_data, hide_index=True, use_container_width=True)
        else:
            st.info("💡 尚未有歷史預測數據")
    except Exception as e:
        st.caption(f"準確率數據更新中...")

    st.markdown("---")
    
    # --- 4. 核心指標儀表板 (優化判定邏輯) ---
    st.write("### 📊 核心戰略指標 (Oracle Strategy Metrics)")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        # 活潑度百分比化：讓高價股與低價股有統一標準
        curr_p = safe_float(row[3]) 
        atr_raw = safe_float(row[33]) if len(row) > 33 else 0.0
        atr_pct = (atr_raw / curr_p * 100) if curr_p > 0 else 0.0
        
        atr_desc = "🔥 洗盤劇烈" if atr_pct > 4.5 else "✅ 波動適中" if atr_pct > 2.0 else "💤 走勢平穩"
        st.metric("股價活潑度 (ATR%)", f"{atr_pct:.2f}%")
        st.caption(f"💡 指標：{atr_desc}")

    with col_b:
        # 資金意願修正：放寬門檻，避免微小波動就顯示動能不足
        vol_b = safe_float(row[34]) if len(row) > 34 else 0.0
        if vol_b > 1.2:
            v_status, v_delta = "🔥 資金湧入", "inverse"
        elif vol_b < -1.2:
            v_status, v_delta = "❄️ 動能不足", "normal"
        else:
            v_status, v_delta = "⚖️ 正常換手", "off"
        st.metric("資金追價意願", v_status, delta=f"{vol_b}%", delta_color=v_delta)
        st.caption("💡 正數代表買盤推升力道強勁。")

    with col_c:
        # 性價比修正：下修門檻至 1.2，適應多頭行情
        rr_v = safe_float(row[35]) if len(row) > 35 else 0.0
        rr_txt = "💎 極具價值" if rr_v >= 1.2 else "⚠️ 風險偏高" if rr_v < 0.7 else "📝 空間有限"
        st.metric("投資性價比 (R/R)", rr_txt)
        st.caption(f"💡 風險報酬比：{rr_v:.1f} (1.2以上為佳)")

    st.markdown("---")

    # --- 5. AI 診斷與展望 (保持原樣) ---
    st.write("### 🧠 AI 深度診斷")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info(f"**【AI 臨床診斷】**\n\n{row[27] if len(row) > 27 else '計算中'}")
    with col_d2:
        st.success(f"**【未來展望評估】**\n\n{row[28] if len(row) > 28 else '計算中'}")
 

# ==========================================
# --- 6. Oracle 全維度三層防護翻譯官 (終極避錯版) ---
# ==========================================
    st.markdown("---")
    st.write("### 🧠 Oracle 核心決策指令 (全維度診斷)")

    # 1. 數據提取與預處理 (抓取 AI 大腦產出的核心 10 欄位)
    try:
        # 數據提取：精準對應 38 欄清單
        s_val = row[36] if len(row) > 36 else ""       # AK: sentiment
        m_val = safe_float(row[34]) if len(row) > 34 else 0.0  # AI: vol_bias (資金追價)
        r_val = safe_float(row[35]) if len(row) > 35 else 0.0  # AJ: rr_ratio (性價比)
        
        # 乖離率與當前價
        bias_v = safe_float(row[29]) if len(row) > 29 else 0.0  # AD: bias_5d
        price = safe_float(row[24]) if len(row) > 24 else 0.0   # Y: actual_close (實際價)
        if price == 0: price = safe_float(row[2]) # 若無實際價，用預測價替代
        
        # 支撐與壓力位 (改用預測區間與5日壓力)
        low_bound = safe_float(row[3]) if len(row) > 3 else 0.0  # D: range_low (支撐)
        res_v = safe_float(row[18]) if len(row) > 18 else 9999.0 # S: res_5d (壓力位)
        
    except Exception as e:
        st.error(f"Oracle 數據提取失敗: {e}")
        return

    # 2. 三層防護層狀態判定 (修正判定邏輯)
    # A. 趨勢層：股價站在 AI 預測支撐 (range_low) 之上，且 5 日乖離未過熱 (<8%)
    trend_ok = (price > low_bound) and (bias_v < 8)
    
    # B. 資金層：看 vol_bias (資金追價意願) 是否大於 1.0
    money_ok = (m_val > 1.0)
    
    # C. 空間層：看距離 5 日壓力位 (res_5d) 是否還有 3% 以上獲利空間
    space_ok = ((res_v - price) / price) > 0.03 if price > 0 else False

    # 3. 視覺化紅綠燈顯示
    st.write("#### 🚥 避錯防護網")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📈 股價趨勢", "看漲" if trend_ok else "跌勢", delta=None, delta_color="normal")
    with c2: st.metric("💰 資金動向", "做多" if money_ok else "做空", delta=None, delta_color="normal")
    with c3: st.metric("📏 獲利空間", "利多" if space_ok else "利少", delta=None, delta_color="normal")

    # 4. 細部診斷紀錄 (修正變數名稱以符合 38 欄位定義)
    diag_details = []
    
    # 乖離與支撐診斷 (將 ma20_v 替換為我們剛剛定義的 low_bound)
    if price > low_bound and bias_v > 8:
        diag_details.append("🏃 **衝太快了**：股價短期噴發過猛，乖離偏大，現在衝進去很容易被割在短線高點。")
    elif price < low_bound:
        diag_details.append("🌧️ **還在淋雨**：股價掉在 AI 預測支撐線 (range_low) 下方，上方全是套牢的人，操作難度很高。")

    # 資金動向診斷 (對應 AI 欄 vol_bias)
    if m_val > 3:
        diag_details.append("🔥 **資金湧入**：數據顯示買盤推升力道強勁，屬於積極型進場訊號。")
    elif m_val < -2:
        diag_details.append("🚨 **大戶在溜**：市場情緒看起來不錯，但數據顯示大戶資金正加速撤離，小心變接盤俠。")
    elif m_val > 1 and "恐慌" in s_val:
        diag_details.append("👀 **黃金背離**：市場氣氛恐慌，但 AI 偵測到有聰明錢在低檔偷偷接盤，這是止跌訊號。")

    # 空間壓力診斷 (對應 S 欄 res_5d)
    dist_to_res = ((res_v - price) / price) * 100 if price > 0 else 0
    if dist_to_res < 2 and dist_to_res > 0:
        diag_details.append(f"🧱 **前方撞牆**：距離上方壓力位 ({res_v}) 太近了，獲利空間不到 2%，這筆交易不划算。")

    # 5. Oracle 最終裁決邏輯 (綜合診斷結果)
    st.markdown("---")
    if trend_ok and money_ok and space_ok:
        status, icon, color = "💎 絕佳擊球點", "✅", "success"
        advice = "目前萬事俱備！數據顯示這是高品質的起漲訊號。趨勢、資金與空間形成共鳴，是避開錯誤後的最佳進場點。"
    
    elif any("🚨" in s or "🧐" in s for s in diag_details):
        status, icon, color = "🚫 避開致命陷阱", "🛑", "error"
        advice = "數據中藏著極高風險！可能是假突破或大戶正在倒貨給散戶。Oracle 建議：忍住誘惑，空手觀望。"
    
    elif "🌧️" in str(diag_details) and "👀" in str(diag_details):
        status, icon, color = "⏳ 潛力底部分批買", "🟡", "warning"
        advice = "雖然大趨勢還沒轉強，但已經看到法人低位撿便宜的影子。適合長線投資者開始小量建立基本持股。"
    
    elif not space_ok and trend_ok:
        status, icon, color = "🚧 空間受限，先看戲", "⚠️", "warning"
        advice = "雖然趨勢是對的，但現在買進就像在天花板下跳舞，賺不到錢。建議等股價突破壓力位站穩後再進場。"
    
    else:
        status, icon, color = "⚖️ 觀望為宜", "📝", "info"
        advice = "目前的訊號混亂，沒有明確的贏面。避開錯誤的最佳方式就是『不看不懂的盤』，建議把資金留在更有活力的目標。"

    # 6. 漂亮結果輸出
    st.markdown(f"#### {icon} {status}")
    
    # 輸出條列式細節
    for item in diag_details:
        st.write(f"- {item}")
    
    # 輸出總結建議框
    if color == "success": st.success(f"**Oracle 總結建議：** {advice}")
    elif color == "warning": st.warning(f"**Oracle 總結建議：** {advice}")
    elif color == "error": st.error(f"**Oracle 總結建議：** {advice}")
    else: st.info(f"**Oracle 總結建議：** {advice}")

# ==========================================
# 第七章：AI 戰略委員會 (穩定路徑最終版)
# ==========================================
def chapter_7_ai_committee_analysis(symbol, brain_row):
    st.markdown("---")
    st.write("### 🎖️ AI 戰略委員會 (全指標對撞診斷)")

    # 1. 嚴格權限檢查 (只允許 admin)
    user_val = ""
    # 增加更多可能的 Key 檢查，確保穩定抓到 admin
    for k in ["username", "user_id", "user", "name", "login_user"]:
        if k in st.session_state and st.session_state[k]:
            if str(st.session_state[k]).strip().lower() == "admin":
                user_val = "admin"
                break

    if user_val != "admin":
        st.info("🔒 此功能為『系統管理員 admin』專屬。")
        return

    # 2. 數據預處理
    full_brain_data = ", ".join([str(item) for item in brain_row]) 
    analysis_task = f"你是首席戰略官。請分析股票 {symbol}。量化指標：{full_brain_data}。請給出投資建議。"

    # 3. 按鈕啟動
    if st.button("🚀 啟動診斷：召開軍師會議", key="gem_admin_final_fix", type="primary", use_container_width=True):
        with st.spinner(f"管理員 admin 您好，AI 軍師正在強制切換穩定路徑..."):
            import google.generativeai as genai
            
            # 配置 API
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            # 💡 核心修正：使用穩定版名稱，避開 v1beta 找不到 gemini-pro 的問題
            # 同時完全移除工具調用 (tools)，確保不會噴 Unknown field
            models_to_try = [
                "gemini-1.5-flash-latest", # 目前最穩定的全版本通用名稱
                "gemini-1.5-flash", 
                "models/gemini-1.5-flash"
            ]
            
            success = False
            last_err = ""
            
            for m_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name=m_name)
                    response = model.generate_content(analysis_task)
                    
                    if response and response.text:
                        st.markdown(f"#### 🗨️ {symbol} 戰略報告")
                        st.markdown(response.text)
                        st.success(f"✅ 診斷完成 (路徑: {m_name})")
                        success = True
                        break
                except Exception as e:
                    last_err = str(e)
                    continue
            
            if not success:
                st.error(f"🚨 API 調用失敗。錯誤訊息：{last_err}")
                st.info("💡 提示：請確認您的 API Key 是否在 Google AI Studio 中正確啟用，且沒有超過免費層級限制。")
                    
# 確保程式啟動
if __name__ == "__main__":
    main()




