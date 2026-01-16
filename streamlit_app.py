import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import gspread
from google.oauth2.service_account import Credentials
import time

# =================================================================
# 1. 高對比度與亮色視覺設定
# =================================================================
st.set_page_config(layout="wide", page_title="Oracle AI Terminal")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    /* 強制所有文字變亮白色 */
    p, span, label, .stMetric label { color: #FFFFFF !important; font-weight: 500 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 28px !important; }
    
    /* 漲跌標示 */
    .price-up { color: #FF4B4B !important; font-weight: bold; font-size: 24px; } 
    .price-down { color: #00E676 !important; font-weight: bold; font-size: 24px; } 
    
    /* 區塊容器 */
    .ai-card {
        background-color: #1A1A1A;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 15px;
    }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #333; color: white; border: 1px solid #666; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. 資料庫連線 (保持原邏輯)
# =================================================================
@st.cache_resource
def get_db():
    creds_info = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_info: return None
    try:
        info = json.loads(creds_info)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(info, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open("users")
        return {
            "user_ws": sh.worksheet("users"),
            "watch_ws": sh.worksheet("watchlist"),
            "pred_ws": sh.worksheet("predictions")
        }
    except: return None

# =================================================================
# 3. 無側邊欄主程式 (手機優先)
# =================================================================
def main_app(db):
    # --- 頂部管理區 (取代側邊欄) ---
    st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>🔮 ORACLE AI 終端</h2>", unsafe_allow_html=True)
    
    # 用戶資訊與登出
    top_c1, top_c2 = st.columns([3, 1])
    top_c1.write(f"👤 用戶: **{st.session_state['user']}**")
    if top_c2.button("🚪 登出"):
        st.session_state["logged_in"] = False
        st.rerun()

    # 清單管理
    watch_data = db["watch_ws"].get_all_values()
    my_stocks = [r[1] for r in watch_data if r[0] == st.session_state['user']]
    
    # 20 支限制與新增
    st.markdown(f"**📈 監控清單 ({len(my_stocks)}/20)**")
    add_col1, add_col2 = st.columns([3, 1])
    
    if len(my_stocks) < 20:
        new_s = add_col1.text_input("輸入新代碼 (例: 2330.TW)", key="new_s").strip().upper()
        if add_col2.button("✚ 新增"):
            if new_s and new_s not in my_stocks:
                db["watch_ws"].append_row([st.session_state['user'], new_s])
                st.rerun()
    else:
        st.warning("⚠️ 清單已達 20 支上限")

    # 選擇股票
    target = st.selectbox("🎯 選擇觀測標的", ["請選擇"] + my_stocks, label_visibility="collapsed")
    
    if target == "請選擇":
        st.info("請選擇上方股票開始分析")
        return

    st.divider()

    # --- 數據讀取與格式容錯 (解決 KeyError) ---
    raw_preds = db["pred_ws"].get_all_values()
    if len(raw_preds) > 1:
        # 強制轉小寫標題並搜尋
        headers = [h.strip().lower() for h in raw_preds[0]]
        df_p = pd.DataFrame(raw_preds[1:], columns=headers)
        # 匹配 symbol 欄位 (支援大小寫容錯)
        stock_pred = df_p[df_p['symbol'].str.upper() == target.upper()].tail(1)
    else:
        stock_pred = pd.DataFrame()

    # 抓取即時報價
    with st.spinner("同步市場報價..."):
        tk = yf.Ticker(target)
        h = tk.history(period="5d")
        if h.empty:
            st.error("找不到市場數據，請確認代碼 (台股需含 .TW)")
            return
        curr = h['Close'].iloc[-1]
        diff = curr - h['Close'].iloc[-2]
        pct = (diff / h['Close'].iloc[-2]) * 100

    # 報價看板
    c_up = diff >= 0
    st.markdown(f"""
        <div style='text-align:center; padding:10px;'>
            <div style='font-size:16px;'>{target} 當前報價</div>
            <div class="{'price-up' if c_up else 'price-down'}">{curr:.2f} ({diff:+.2f} / {pct:+.2f}%)</div>
        </div>
    """, unsafe_allow_html=True)

    # --- 核心顯示區 ---
    if not stock_pred.empty:
        row = stock_pred.iloc[0].to_dict()
        
        # 1. AI 診斷 (AB, AC)
        st.markdown(f"<div class='ai-card' style='border-left: 5px solid #FF4B4B;'><b>🔍 AI 診斷 (AB)</b><br>{row.get('ai_insight', '資料庫欄位缺失')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-card' style='border-left: 5px solid #00E676;'><b>🔮 展望目標 (AC)</b><br>{row.get('forecast_outlook', '資料庫欄位缺失')}</div>", unsafe_allow_html=True)
        
        # 2. 戰略水位矩陣
        st.markdown("### 🛡️ 戰略水位 (G-X)")
        l1, l2, l3 = st.columns(3)
        l1.metric("支撐位", row.get('buy_level_5d', '--'))
        l2.metric("目標價", row.get('sell_level_5d', '--'))
        l3.metric("強壓位", row.get('resist_level_5d', '--'))

        # 3. 手動更新按鈕 (即便有資料，也放在下方供隨時手動分析)
        if st.button("🔄 立即重新執行 AI 深度分析"):
            run_manual_analysis(target, db)
    else:
        # --- 沒資料時顯示手動按鈕 ---
        st.warning(f"⚠️ 標的 {target} 目前尚無預測資料")
        if st.button("🚀 啟動 Oracle AI 進行首次分析"):
            run_manual_analysis(target, db)

# =================================================================
# 4. 手動分析執行 (與 cron_job 對接)
# =================================================================
def run_manual_analysis(symbol, db):
    with st.spinner(f"Oracle AI 正在為 {symbol} 進行 800 次模擬運算..."):
        try:
            from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context
            # 抓取大腦所需資料
            df, final_id = fetch_comprehensive_data(symbol)
            mkt = fetch_market_context()
            # 運算
            p_val, p_path, p_diag, p_out, p_bias, p_levels, p_experts = god_mode_engine(df, final_id, mkt)
            
            # 打包 37 欄位寫入 (確保欄位順序對齊 A-AK)
            row_to_add = [datetime.now().strftime("%Y-%m-%d"), final_id, p_val, 0, 0, "手動更新"] + p_levels + [0, 0, p_path, p_diag, p_out] + p_bias + p_experts
            db["pred_ws"].append_row(row_to_add)
            
            st.success("分析完成！")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"分析失敗: {e}")

# =================================================================
# 5. 認證與入口 (保持原邏輯但優化顏色)
# =================================================================
def auth_section(db):
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔮 ORACLE AI LOGIN</h1>", unsafe_allow_html=True)
    u = st.text_input("帳號 (Username)").strip()
    p = st.text_input("密碼 (Password)", type="password").strip()
    if st.button("解鎖終端"):
        raw_users = db["user_ws"].get_all_values()
        # 兼容標題列檢查
        users = raw_users[1:] if len(raw_users) > 0 else []
        found = next((r for r in users if r[0] == u and r[1] == p), None)
        if found:
            st.session_state["logged_in"] = True
            st.session_state["user"] = u
            st.rerun()
        else:
            st.error("認證失敗：帳號或密碼不匹配")

if __name__ == "__main__":
    db_conn = get_db()
    if db_conn:
        if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
        if not st.session_state["logged_in"]:
            auth_section(db_conn)
        else:
            main_app(db_conn)
