import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. 配置與專業視覺優化 (包含輸入欄文字強制白色修正) ---
st.set_page_config(page_title="StockAI 全能技術終端", layout="wide")

st.markdown("""
    <style>
    /* 全域背景與文字基礎 */
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 600 !important; }
    
    /* 隱藏側邊欄按鈕，強制置頂面板 */
    button[data-testid="sidebar-button"] { display: none !important; }

    /* 下拉選單、輸入框視覺優化 */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { 
        background-color: #1C2128 !important; 
        border: 2px solid #00F5FF !important; 
        border-radius: 10px !important;
    }

    /* 【核心修正】強制輸入欄位內的文字為白色，解決黑色背景看不見的問題 */
    input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important; /* 確保手機 Safari 也顯示白色 */
    }
    
    /* 數字輸入框控制按鈕 (+/-) 顏色 */
    button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] {
        background-color: #30363D !important;
        color: #FFFFFF !important;
    }

    /* 儀表板卡片 */
    [data-testid="stMetricValue"] { color: #00F5FF !important; font-weight: bold; font-size: 2.2rem !important; }
    .stMetric { background-color: #1C2128; border: 2px solid #30363D; border-radius: 15px; padding: 20px; }
    
    /* 置頂管理面板樣式 */
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #444C56 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
    
    /* 亮青色大按鈕樣式 */
    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border: none !important;
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important;
        width: 100% !important;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #FFFFFF !important; box-shadow: 0px 0px 15px #00F5FF; }
    
    /* 診斷卡片 */
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 18px; margin-bottom: 12px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (保留原始版本 3 次重試與所有技術指標) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    for _ in range(3):
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=True, repair=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 保留所有技術指標計算
                data['MA5'] = data['Close'].rolling(5).mean()
                data['MA20'] = data['Close'].rolling(20).mean()
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Hist'] = data['MACD'] - data['Signal']
                return data.dropna()
            time.sleep(1.5)
        except:
            time.sleep(1.5); continue
    return None

# --- 3. AI 診斷分析模組 ---
def perform_ai_analysis(df, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 50
    reasons = []
    
    # 技術因子判斷
    if last['Close'] > last['MA20']:
        score += 15; reasons.append("🟢 **多頭趨勢**: 股價站穩月線，支撐力道扎實。")
    else:
        score -= 10; reasons.append("🔴 **趨勢偏弱**: 股價低於月線，短期動能承壓。")
        
    if last['Hist'] > prev['Hist']:
        score += 10; reasons.append("🔥 **買氣增溫**: MACD 紅色能量柱持續放大中。")
    
    # 情感分析模擬
    news = [
        {"tag": "利多", "content": "產業需求展望優於預期，法人維持買進評等", "val": 10},
        {"tag": "中性", "content": "短期市場情緒震盪，關注聯準會最新動向", "val": -2}
    ]
    
    final_score = max(0, min(100, score + sum(n['val'] for n in news) + (int(precision)-55)))
    return int(final_score), reasons, news

# --- 4. 繪圖與儀表板核心展示 ---
def show_ultimate_dashboard(symbol, unit, p_days, precision):
    df = fetch_comprehensive_data(symbol)
    if df is None:
        st.error(f"❌ 無法讀取股票代碼 '{symbol}'。"); return

    # AI 預測邏輯 (保留 Monte Carlo 模擬細節)
    last_p = float(df['Close'].iloc[-1])
    noise = np.random.normal(0, 0.002, p_days)
    trend = (int(precision) - 55) / 500
    pred_prices = last_p * np.cumprod(1 + trend + noise)
    
    ai_score, ai_reasons, ai_news = perform_ai_analysis(df, precision)

    # 頂部卡片
    target_p = pred_prices[-1]
    pct = ((target_p - last_p)/last_p)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", f"{last_p:.2f}")
    c2.metric(f"AI 預估({p_days}天)", f"{target_p:.2f}")
    c3.metric("預期回報", f"{pct:.2f}%", delta=f"{pct:.2f}%")

    # AI 診斷報告區塊
    st.divider()
    d_col1, d_col2 = st.columns([1, 1.2])
    with d_col1:
        st.markdown(f"### 💡 AI 技術因子診斷 (評分: `{ai_score}`)")
        for r in ai_reasons: st.write(r)
    with d_col2:
        st.markdown("### 📰 市場情感標籤")
        for n in ai_news:
            st.markdown(f"<div class='diag-box'><b>[{n['tag']}]</b> {n['content']}</div>", unsafe_allow_html=True)

    # Plotly 專業圖表配置 (救回 K線顏色區分與 MACD 歷史細節)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.3], vertical_spacing=0.04)
    zoom = {"日": 45, "月": 180, "年": 550}[unit]
    p_df = df.tail(zoom)
    
    # 1. K線、MA5、MA20
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線', increasing_line_color='#00FF41', decreasing_line_color='#FF3131'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=2.5)), row=1, col=1)
    
    # 2. AI 預測路徑
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scattergl(x=f_dates, y=pred_prices, name='AI 預測', line=dict(color='#FF4500', width=4.5, dash='dashdot')), row=1, col=1)
    
    # 3. 成交量 (救回顏色區分邏輯)
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors, opacity=0.8), row=2, col=1)
    
    # 4. MACD (救回歷史柱顏色區分邏輯)
    h_colors = ['#FF3131' if v < 0 else '#00FF41' for v in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD力道', marker_color=h_colors), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.info(f"📊 **總結說明**：{symbol} 目前 AI 評分為 {ai_score}。技術面顯示 {ai_reasons[0][4:]}。綜合新聞面之情感反應，建議投資者關注後續成交量是否能有效放大。")

# --- 5. 主程式 (保留所有 Sheet 連線與管理員控制) ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if 'last_sync' not in st.session_state: st.session_state.last_sync = datetime.now()

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_user, ws_watch, ws_settings = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except:
        st.error("🚨 數據庫安全連線失敗。"); return

    # 讀取全域設定
    s_data = {item['setting_name']: item['value'] for item in ws_settings.get_all_records()}
    curr_prec, curr_ttl = int(s_data.get('global_precision', 55)), int(s_data.get('api_ttl_min', 5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 全能技術終端")
        t1, t2 = st.tabs(["🔑 系統登入", "📝 快速註冊"])
        with t1:
            u, p = st.text_input("帳號", key="login_u"), st.text_input("密碼", type="password", key="login_p")
            if st.button("確認登入", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with t2:
            nu, npw = st.text_input("新帳號"), st.text_input("新密碼", type="password")
            if st.button("註冊帳號", use_container_width=True):
                udf = pd.DataFrame(ws_user.get_all_records())
                if nu and nu not in udf['username'].astype(str).tolist():
                    ws_user.append_row([nu, npw]); st.success("註冊成功！")
                else: st.error("帳號已存在")
    else:
        # --- 頂部面板 (取代側邊欄，解決手機操作難題) ---
        remain = (st.session_state.last_sync + timedelta(minutes=curr_ttl)) - datetime.now()
        st.markdown(f"👤 **{st.session_state.user}** | 🕒 刷新倒數: **{max(0, int(remain.total_seconds()))}s**")
        
        with st.expander("⚙️ 終端功能面板 (管理清單與系統參數)", expanded=False):
            m1, m2 = st.columns([1, 1])
            with m1:
                st.subheader("📋 清單管理")
                all_w = pd.DataFrame(ws_watch.get_all_records())
                user_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("我的選股清單", user_stocks if user_stocks else ["2330.TW"])
                
                # 保留刪除邏輯
                if user_stocks and st.button(f"🗑️ 刪除 {target}"):
                    rows = ws_watch.get_all_values()
                    for i, row in enumerate(rows):
                        if i > 0 and row[0] == st.session_state.user and row[1] == target:
                            ws_watch.delete_rows(i + 1); st.rerun()
                
                # 保留新增邏輯
                ns = st.text_input("➕ 新增股票代碼 (例: AAPL)").strip().upper()
                if st.button("確認新增"):
                    if ns and ns not in user_stocks:
                        ws_watch.append_row([st.session_state.user, ns]); st.rerun()
            
            with m2:
                st.subheader("🛠️ 參數與連線")
                # 保留 okdycrreoo 管理員控制邏輯
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    new_t = st.select_slider("快取連線時間 (分)", options=list(range(1, 11)), value=curr_ttl)
                    if st.button("💾 同步資料庫設定"):
                        ws_settings.update_cell(2, 2, str(new_p))
                        ws_settings.update_cell(3, 2, str(new_t))
                        st.cache_data.clear()
                        st.session_state.last_sync = datetime.now()
                        st.rerun()
                else:
                    st.info(f"靈敏度目前由管理員統一設定為: {curr_prec}")
                
                unit = st.selectbox("時間單位", ["日", "月", "年"])
                p_days = st.number_input("AI 預測天數", 1, 30, 7)
                if st.button("🚪 安全登出系統"): st.session_state.user = None; st.rerun()

        # 執行主要顯示
        show_ultimate_dashboard(target, unit, p_days, curr_prec)

if __name__ == "__main__":
    main()
