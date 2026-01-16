import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz

# =================================================================
# 段落 1：核心引擎加載 (修正縮進錯誤)
# =================================================================
try:
    # 這裡前面必須有 4 個空格 (縮進)
    from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context, init_gspread
    engine_available = True
except Exception as e:
    # 這裡前面也必須有 4 個空格
    st.error(f"⚠️ 引擎加載失敗，請檢查 cron_job.py 位置是否正確。錯誤: {e}")
    engine_available = False

# =================================================================
# 段落 2：資料庫連線 (使用現代化 google-auth)
# =================================================================
@st.cache_resource
def get_db():
    # 從 Streamlit Secrets 讀取憑證
    creds_info = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_info:
        st.error("❌ 請在 Streamlit Secrets 設定 GCP_SERVICE_ACCOUNT_JSON")
        return None
    
    try:
        # 解析 JSON
        info = json.loads(creds_info)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        
        # 建立連線
        creds = Credentials.from_service_account_info(info, scopes=scope)
        client = gspread.authorize(creds)
        
        # 開啟試算表
        sh = client.open("users")
        return {
            "user_ws": sh.worksheet("users"),
            "watch_ws": sh.worksheet("watchlist"),
            "pred_ws": sh.worksheet("predictions")
        }
    except Exception as e:
        st.error(f"❌ 連線 Google Sheets 失敗，請檢查 JSON 格式或權限: {e}")
        return None
# =================================================================
# 段落 3：會員系統 (兼容您現有的 users 表格)
# =================================================================
def auth_section(db):
    st.title("🔮 Oracle AI 終端")
    tab1, tab2 = st.tabs(["登入系統", "註冊帳號"])
    
    with tab1:
        u = st.text_input("帳號", key="login_u")
        p = st.text_input("密碼", type="password", key="login_p")
        if st.button("立即進入"):
            users = db["user_ws"].get_all_records()
            found = next((row for row in users if str(row['username']) == u and str(row['password']) == p), None)
            if found:
                st.session_state["logged_in"] = True
                st.session_state["user"] = u
                st.rerun()
            else:
                st.error("帳號或密碼錯誤")

    with tab2:
        new_u = st.text_input("設定帳號", key="reg_u")
        new_p = st.text_input("設定密碼", type="password", key="reg_p")
        if st.button("確認註冊"):
            users = db["user_ws"].get_all_records()
            if any(str(row['username']) == new_u for row in users):
                st.warning("此帳號已被使用")
            elif new_u and new_p:
                db["user_ws"].append_row([new_u, new_p])
                st.success("註冊成功！請切換至登入分頁。")
            else:
                st.error("欄位不可為空")

# =================================================================
# 段落 4：主功能介面 (手機直向優化)
# =================================================================
def main_app(db):
    # --- 頂部導航與登出 ---
    t1, t2 = st.columns([3, 1])
    t1.subheader(f"👋 歡迎, {st.session_state['user']}")
    if t2.button("🚪 登出系統"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.divider()

    # --- 1. 新增股票區塊 (含 20 支限制提醒) ---
    all_watch = db["watch_ws"].get_all_records()
    my_stocks = [r['symbol'] for r in all_watch if str(r['username']) == st.session_state['user']]
    stock_count = len(my_stocks)

    with st.expander("➕ 管理我的觀測清單", expanded=False):
        # 顯示當前數量提醒
        if stock_count >= 20:
            st.error(f"⚠️ 已達上限：目前的清單已有 {stock_count}/20 支股票，請刪除舊標的再新增。")
        else:
            st.info(f"💡 目前清單：{stock_count}/20 (上限 20 支)")
            new_s = st.text_input("輸入股票代碼 (例如: 2330, NVDA)", key="add_s").strip().upper()
            if st.button("確認新增"):
                if new_s and new_s not in my_stocks:
                    db["watch_ws"].append_row([st.session_state['user'], new_s])
                    st.success(f"✅ {new_s} 已加入清單！")
                    st.rerun()

    # --- 2. 選擇個股與診斷 ---
    if not my_stocks:
        st.info("您的清單目前為空，請先在上方新增股票。")
        return

    target = st.selectbox("🎯 選擇觀測個股", ["請選擇"] + my_stocks)

    if target != "請選擇":
        all_preds = db["pred_ws"].get_all_records()
        df_p = pd.DataFrame(all_preds)
        
        # 過濾該股最新一筆數據
        stock_data = pd.DataFrame()
        if not df_p.empty and 'symbol' in df_p.columns:
            stock_data = df_p[df_p['symbol'].str.contains(target, na=False)].tail(1)

        if stock_data.empty:
            st.warning(f"目前尚無 {target} 的分析數據")
            if st.button(f"🚀 啟動即時 AI 診斷"):
                with st.spinner(f"正在為 {target} 執行預測之神引擎分析..."):
                    try:
                        # 1. 抓取數據
                        mkt_df = fetch_market_context()
                        df, f_id = fetch_comprehensive_data(target)
                        
                        if df is not None:
                            # 2. 執行 AI 核心運算
                            p_next, path_str, insight, biases, s_data, e_data = god_mode_engine(df, f_id, mkt_df)
                            
                            # 3. 準備寫入 Google Sheets 的數據列 (對齊 35 欄格式)
                            data_date = df.index[-1].strftime("%Y-%m-%d")
                            # s_data 包含 5, 10, 15, 20, 25, 30 日的數據，我們取前段
                            upload_row = [
                                data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "即時更新"
                            ] + s_data + [0] + [path_str, insight] + biases + e_data
                            
                            # 4. 寫入試算表
                            db["pred_ws"].append_row(upload_row)
                            
                            st.success(f"✅ {target} 診斷完成！數據已同步至雲端。")
                            time.sleep(1)
                            st.rerun() # 強制刷新頁面以顯示新數據
                        else:
                            st.error("無法從 Yahoo Finance 獲取該股票數據，請檢查代碼是否正確。")
                    except Exception as e:
                        st.error(f"❌ 診斷失敗：{str(e)}")
            
            # --- AI 關鍵診斷報告 ---
            st.success(f"🤖 **AI 診斷報告：**\n\n{row.get('ai_insight', '無報告')}")

            # --- 核心支撐與壓力戰術板 (5D, 10D, 20D) ---
            st.markdown("### 🛡️ AI 戰術水位線 (買賣點參考)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**5日 (短線)**")
                st.write(f"⬆️ 壓力: `{row.get('sell_level_5d', 'N/A')}`")
                st.write(f"⬇️ 買入: `{row.get('buy_level_5d', 'N/A')}`")

            with col2:
                st.warning("**10日 (週線)**")
                st.write(f"⬆️ 壓力: `{row.get('sell_level_10d', 'N/A')}`")
                st.write(f"⬇️ 買入: `{row.get('buy_level_10d', 'N/A')}`")

            with col3:
                st.error("**20日 (月線)**")
                st.write(f"⬆️ 壓力: `{row.get('sell_level_20d', 'N/A')}`")
                st.write(f"⬇️ 買入: `{row.get('buy_level_20d', 'N/A')}`")

            # --- 預測走勢圖 ---
            st.markdown("### 📈 未來 7 日模擬軌跡")
            path_vals = [float(x) for x in str(row.get('pred_path', '0')).split(',')]
            st.line_chart(path_vals)

# =================================================================
# 段落 5：主入口
# =================================================================
if __name__ == "__main__":
    db_con = get_db()
    if db_con:
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        
        if not st.session_state["logged_in"]:
            auth_section(db_con)
        else:
            main_app(db_con)






