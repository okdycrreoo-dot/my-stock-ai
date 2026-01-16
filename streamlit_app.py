import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import gspread
from google.oauth2.service_account import Credentials
import time

# =================================================================
# 1. 高對比度與亮色視覺 (九大項目要求)
# =================================================================
st.set_page_config(layout="wide", page_title="Oracle AI Terminal")
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    p, span, label, h1, h2, h3 { color: #FFFFFF !important; }
    .price-up { color: #FF3131 !important; font-weight: bold; }
    .price-down { color: #00FF00 !important; font-weight: bold; }
    .ai-box { padding: 15px; border-radius: 10px; border: 1px solid #333; background-color: #0A0A0A; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. 資料庫連線
# =================================================================
@st.cache_resource
def get_db():
    try:
        creds_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open("users")
        return {"user_ws": sh.worksheet("users"), "watch_ws": sh.worksheet("watchlist"), "pred_ws": sh.worksheet("predictions")}
    except: return None

# =================================================================
# 3. 主程式 - 九大項目實現
# =================================================================
def main_app(db):
    # --- A. 摺疊式管理面板 (需求 2) ---
    with st.expander("🛠️ 管理監控清單 (點擊展開/縮放)", expanded=False):
        watch_data = db["watch_ws"].get_all_values()
        # 取得當前用戶清單 (排除標題並去空格)
        my_stocks = [r[1].strip() for r in watch_data if r[0] == st.session_state['user']]
        
        c1, c2 = st.columns(2)
        with c1:
            if len(my_stocks) < 20:
                new_s = st.text_input("✚ 新增代碼", key="add_s").strip().upper()
                if st.button("確認新增"):
                    if new_s and new_s not in my_stocks:
                        db["watch_ws"].append_row([st.session_state['user'], new_s])
                        st.rerun()
            else: st.error("清單已達 20 支上限")
        
        with c2:
            del_s = st.selectbox("🗑️ 刪除股票 (需求 1)", ["選擇刪除標的"] + my_stocks)
            if st.button("執行刪除") and del_s != "選擇刪除標的":
                # 找到對應行號刪除
                for i, r in enumerate(watch_data):
                    if r[0] == st.session_state['user'] and r[1] == del_s:
                        db["watch_ws"].delete_rows(i + 1)
                        st.rerun()
        
        if st.button("🚪 登出系統"):
            st.session_state["logged_in"] = False
            st.rerun()

    # --- B. 標的選擇與數據處理 ---
    target = st.selectbox("🎯 選擇觀測標的", ["請選擇"] + my_stocks)
    if target == "請選擇": return

    # 讀取預測數據並清洗 (排除標題)
    raw_p = db["pred_ws"].get_all_values()
    df_p = pd.DataFrame(raw_p[1:], columns=raw_p[0]) if len(raw_p) > 1 else pd.DataFrame()
    # 確保 symbol 比對一致
    stock_pred = df_p[df_p['symbol'].str.strip() == target].tail(1) if not df_p.empty else pd.DataFrame()

    # --- C. 九大項目展示 ---
    # 項目 1, 9: 即時報價與漲跌
    tk = yf.Ticker(target)
    hist = tk.history(period="60d")
    if hist.empty:
        st.error("代碼錯誤或無數據")
        return
    
    curr = hist['Close'].iloc[-1]
    diff = curr - hist['Close'].iloc[-2]
    pct = (diff / hist['Close'].iloc[-2]) * 100
    color = "price-up" if diff >= 0 else "price-down"
    
    st.markdown(f"### {target} <span class='{color}'>{curr:.2f} ({diff:+.2f} / {pct:+.2f}%)</span>", unsafe_allow_html=True)

    if not stock_pred.empty:
        row = stock_pred.iloc[0].to_dict()
        
        # 項目 2, 3: AI 診斷與展望 (AB, AC 欄位)
        col_ab, col_ac = st.columns(2)
        with col_ab:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #FF3131;'><b>🔍 AI 綜合診斷 (AB)</b><br>{row.get('ai_insight','計算中...')}</div>", unsafe_allow_html=True)
        with col_ac:
            st.markdown(f"<div class='ai-box' style='border-left: 5px solid #00FFFF;'><b>🔮 未來操作展望 (AC)</b><br>{row.get('forecast_outlook','計算中...')}</div>", unsafe_allow_html=True)

        # 項目 4-8: 專業圖表 (K線, MA, 成交量, MACD, KDJ, AI路徑)
        st.markdown("### 📈 終端指標全圖")
        
        # 指標計算
        h = hist.copy()
        h['MA5'] = h['Close'].rolling(5).mean()
        h['MA20'] = h['Close'].rolling(20).mean()
        exp12 = h['Close'].ewm(span=12, adjust=False).mean()
        exp26 = h['Close'].ewm(span=26, adjust=False).mean()
        h['MACD'] = exp12 - exp26
        h['Signal'] = h['MACD'].ewm(span=9, adjust=False).mean()
        h['Hist'] = h['MACD'] - h['Signal']
        
        low_9 = h['Low'].rolling(9).min()
        high_9 = h['High'].rolling(9).max()
        h['K'] = ((h['Close'] - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean()
        h['D'] = h['K'].ewm(com=2).mean()

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
        
        # K線 + MA
        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['MA5'], name="MA5", line=dict(color='#FFD700')), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['MA20'], name="MA20", line=dict(color='#00FFFF')), row=1, col=1)
        
        # 項目 4: AI 預測路徑 (AA)
        if row.get('pred_path'):
            try:
                pp = [float(x) for x in str(row['pred_path']).split(',')]
                p_dates = [h.index[-1] + timedelta(days=i) for i in range(1, 8)]
                fig.add_trace(go.Scatter(x=p_dates, y=pp, name="AI預測", line=dict(color='#FF3131', dash='dash')), row=1, col=1)
            except: pass

        # 成交量, MACD, KDJ
        fig.add_trace(go.Bar(x=h.index, y=h['Volume'], name="成交量", marker_color='#444'), row=2, col=1)
        fig.add_trace(go.Bar(x=h.index, y=h['Hist'], name="MACD柱"), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['K'], name="K", line=dict(color='white')), row=4, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['D'], name="D", line=dict(color='yellow')), row=4, col=1)

        fig.update_layout(template="plotly_dark", height=900, paper_bgcolor='black', plot_bgcolor='black', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 項目: 戰略水位矩陣與專家指標 (G-X, AH-AK)
        st.markdown("### 🛡️ 戰略水位 (G-X)")
        col1, col2, col3 = st.columns(3)
        # 修正百分比顯示問題，確保數值乾淨
        def fmt(v): return str(v).replace('%','')
        col1.metric("短線支撐", fmt(row.get('buy_5d','--')))
        col2.metric("目標賣點", fmt(row.get('sell_5d','--')))
        col3.metric("強壓關卡", fmt(row.get('resist_5d','--')))

    # --- D. 補回更新按鈕 (需求 3: 防止重複寫入) ---
    if st.button("🚀 執行深度分析 (手動更新)"):
        with st.spinner("AI 運算中..."):
            run_manual_analysis(target, db)

# =================================================================
# 4. 手動分析邏輯 (防止重複)
# =================================================================
def run_manual_analysis(symbol, db):
    try:
        from cron_job import fetch_comprehensive_data, god_mode_engine, fetch_market_context
        df, final_id = fetch_comprehensive_data(symbol)
        mkt = fetch_market_context()
        p_val, p_path, p_diag, p_out, p_bias, p_levels, p_experts = god_mode_engine(df, final_id, mkt)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 防止重複 (需求 3)：檢查是否已有今日同股票資料，有的話先刪除舊的
        all_p = db["pred_ws"].get_all_values()
        for i, r in enumerate(all_p):
            if r[0] == today and r[1] == symbol:
                db["pred_ws"].delete_rows(i + 1)
                break
        
        # 寫入 37 欄位
        row = [today, symbol, p_val, 0, 0, "手動更新"] + p_levels + [0, 0] + [p_path, p_diag, p_out] + p_bias + p_experts
        db["pred_ws"].append_row(row)
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

