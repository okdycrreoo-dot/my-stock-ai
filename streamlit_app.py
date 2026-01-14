import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import time
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 鎖定台灣時區 (確保 14:30 更新精準)
tw_tz = pytz.timezone('Asia/Taipei')

st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    input { color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 600 !important; }
    div[data-baseweb="input"] { background-color: #FFFFFF !important; border-radius: 8px; }
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #000000 !important; border: 2px solid #00F5FF !important; }
    div[role="listbox"] div { color: #000000 !important; }
    .stButton>button { background-color: #00F5FF !important; color: #0E1117 !important; border-radius: 12px; font-weight: 900 !important; height: 3.5rem !important; width: 100% !important; }
    .diag-box { background: #050505; padding: 15px; border-radius: 12px; border: 1px solid #444; min-height: 120px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .ai-advice-box { background: #000000; border: 2px solid #333; padding: 20px; border-radius: 15px; margin-top: 25px; border-left: 10px solid #FFAC33; position: relative; }
    .confidence-tag { background: #FF3131; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_comprehensive_data(symbol, ttl_seconds):
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): s = f"{s}.TW"
    try:
        # 加入 timeout=10 防止 API 阻塞黑屏
        df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False, timeout=10)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            # 保留 MA, MACD, KDJ, RSI, ATR 運算 (同原版)
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
            df['MACD'], df['Signal'] = e12 - e26, (e12 - e26).ewm(span=9).mean()
            df['Hist'] = df['MACD'] - df['Signal']
            l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
            df['K'] = ((df['Close'] - l9) / (h9 - l9 + 0.001) * 100).ewm(com=2).mean()
            df['D'] = df['K'].ewm(com=2).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            delta = df['Close'].diff()
            gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-5))))
            df['ATR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean()
            return df.dropna(), s
    except: pass
    return None, s

def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        now_tw = datetime.now(tw_tz)
        today = now_tw.strftime("%Y-%m-%d")
        
        # 功能：展示最新 10 筆準確率表格 (AI 大腦上方)
        df_acc = df_p[df_p['symbol'] == f_id].tail(10)
        if not df_acc.empty:
            st.markdown("### 🎯 歷史預測準確率追蹤 (最新 10 筆)")
            st.table(df_acc[['date', 'pred_close', 'range_low', 'range_high', 'actual_close', 'error_pct']])

        # 功能：14:30 靜默更新
        if now_tw.hour >= 14 and now_tw.minute >= 30:
            if not any((r['date'] == today and r['symbol'] == f_id) for r in recs):
                ws_p.append_row([today, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""])
        return f"🎯 此股實戰同步完成"
    except: return "🎯 數據同步中"

def auto_fine_tune_engine(df, base_p, base_tw, v_comp, b_list):
    # b_list 為管理員手動輸入的 (藍籌, 成長, ETF)
    try:
        mkt_df = yf.download("^TWII", period="1mo", interval="1d", auto_adjust=True, progress=False, timeout=5)
        mkt_vol = mkt_df['Close'].pct_change().std()
        env_panic = 1.25 if mkt_vol > 0.012 else 1.0
    except: env_panic = 1.0

    rets = df['Close'].pct_change().dropna()
    f_vol = rets.tail(20).std() * env_panic
    f_tw = max(0.5, min(2.5, 1.0 + (rets.tail(5).mean() * 15)))
    
    price_now = float(df['Close'].iloc[-1])
    bias_val = (price_now - df['Close'].rolling(20).mean().iloc[-1]) / (df['Close'].rolling(20).mean().iloc[-1] + 1e-5)
    f_p = (45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60)
    
    b_drift = 0.0
    try:
        b_data = yf.download([f"{c}.TW" if not c.endswith(".TW") else c for c in b_list], period="5d", interval="1d", progress=False, timeout=5)['Close']
        b_drift = b_data.pct_change().iloc[-1].mean()
    except: pass
    
    # 回傳 7 個參數，包含管理員選定的標本 benchmarks
    return int(f_p), round(f_tw, 2), 1.7, b_list, bias_val, f_vol, b_drift

def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol, b_drift):
    # 此處保留您 290 行中所有的 Whale Force, RSI Divergence, Bollinger Squeeze 等 11 項公式
    # ... (省略 50 行公式邏輯，確保回傳 8 個參數) ...
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, insight

def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p, b_list):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: st.error("❌ 讀取失敗"); return

    # 1. 執行 AI 微調與預測
    final_p, final_tw, ai_v, ai_b, bias, f_vol, b_drift = auto_fine_tune_engine(df, cp, tw_val, v_comp, b_list)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, ai_v, bias, f_vol, b_drift)
    
    # 2. 顯示準確率表格 (功能 2)
    stock_accuracy = auto_sync_feedback(ws_p, f_id, insight)

    # 3. 繪製圖表 (保留原版 4 層子圖細節)
    # ... (包含 go.Candlestick, go.Scatter MA, go.Bar Volume, MACD, KDJ 等) ...
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. AI 診斷建議 Box
    st.markdown(f"<div class='ai-advice-box'>...</div>", unsafe_allow_html=True)

def main():
    # ... (初始化 GSheets 連線) ...
    
    if st.session_state.user is None:
        tab_login, tab_reg = st.tabs(["🔑 登入", "📝 註冊"])
        with tab_reg:
            new_u = st.text_input("新帳號", key="reg_u")
            new_p = st.text_input("新密碼", type="password")
            if st.button("提交註冊"):
                udf = pd.DataFrame(ws_u.get_all_records())
                if new_u in udf['username'].astype(str).values: # 功能：註冊防重
                    st.warning("⚠️ 此帳號已存在，請更換名稱")
                else:
                    ws_u.append_row([str(new_u), str(new_p)]); st.success("✅ 註冊成功")
    else:
        # --- 管理員 okdycrreoo 專屬設定 ---
        b_list = [s_map.get('bench_1','2330'), s_map.get('bench_2','2382'), s_map.get('bench_3','00878')]
        if st.session_state.user == "okdycrreoo":
            with st.expander("🛠️ 管理員戰情室", expanded=True):
                # 獲取 AI 核心推薦參數用於比對
                temp_df, _ = fetch_comprehensive_data(target, api_ttl*60)
                ai_p, ai_tw, ai_v, ai_b_rec, _, _, _ = auto_fine_tune_engine(temp_df, cp, tw_val, v_comp, b_list) if temp_df is not None else (cp, tw_val, v_comp, b_list, 0, 0, 0)
                
                # 手動輸入 vs AI 推薦值 (功能 1)
                new_b1 = st.text_input(f"1. 基準藍籌股 (AI推薦: {ai_b_rec[0]})", b_list[0])
                new_b2 = st.text_input(f"2. 成長標本股 (AI推薦: {ai_b_rec[1]})", b_list[1])
                new_b3 = st.text_input(f"3. 指數型 ETF (AI推薦: {ai_b_rec[2]})", b_list[2])
                
                # 保存按鈕更新雲端...


