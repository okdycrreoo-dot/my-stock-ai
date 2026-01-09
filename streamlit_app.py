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

# --- 1. 配置與 UI 視覺 (完整還原所有 CSS 權重，確保行數基準) ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    
    input { 
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important; 
        font-weight: 600 !important; 
    }
    div[data-baseweb="input"] { 
        background-color: #FFFFFF !important; 
        border-radius: 8px; 
    }
    
    div[data-baseweb="select"] > div { 
        background-color: #FFFFFF !important; 
        color: #000000 !important; 
        border: 2px solid #00F5FF !important; 
    }
    div[role="listbox"] div { 
        color: #000000 !important; 
    }

    .stButton>button { 
        background-color: #00F5FF !important; 
        color: #0E1117 !important; 
        border: none !important; 
        border-radius: 12px; 
        font-weight: 900 !important;
        height: 3.5rem !important; 
        width: 100% !important;
    }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; 
        color: #00F5FF !important; 
        border: 2px solid #00F5FF !important; 
        border-radius: 12px !important;
        font-size: 1.2rem !important; 
        font-weight: 900 !important;
    }
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 15px; margin-bottom: 10px; border: 1px solid #30363D; }
    .info-box { background-color: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 10px; text-align: center; min-height: 80px; }
    .ai-advice-box { background-color: #161B22; border: 1px solid #FFAC33; border-radius: 12px; padding: 20px; margin-top: 15px; border-left: 10px solid #FFAC33; }
    .price-buy { color: #FF3131; font-weight: 900; font-size: 1.3rem; }
    .price-sell { color: #00FF41; font-weight: 900; font-size: 1.3rem; }
    .realtime-val { font-size: 1.4rem; font-weight: 900; display: block; margin-top: 5px; }
    .label-text { color: #8899A6 !important; font-size: 0.8rem; letter-spacing: 1px; }
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_data(symbol, ttl_seconds):
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): 
        s = f"{s}.TW"
    for _ in range(3):
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                e12 = df['Close'].ewm(span=12).mean()
                e26 = df['Close'].ewm(span=26).mean()
                df['MACD'] = e12 - e26
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                df['Hist'] = df['MACD'] - df['Signal']
                l9 = df['Low'].rolling(9).min()
                h9 = df['High'].rolling(9).max()
                rsv = (df['Close'] - l9) / (h9 - l9 + 0.001) * 100
                df['K'] = rsv.ewm(com=2).mean()
                df['D'] = df['K'].ewm(com=2).mean()
                df['J'] = 3 * df['K'] - 2 * df['D']
                return df.dropna(), s
            time.sleep(1.5)
        except: 
            time.sleep(1.5)
            continue
    return None, s

# --- 3. AI 核心：深度微調連動引擎 ---
def auto_fine_tune_engine(df, base_p, base_tw, v_comp):
    rets = df['Close'].pct_change().dropna()
    v = rets.tail(20).std()
    adj_p = base_p * (1 + (v * v_comp)) 
    adj_tw = base_tw * (1 + (rets.tail(5).mean() * 12))
    suggested_v = 1.2 if v > 0.03 else 1.8 if v < 0.01 else 1.5
    f_p = max(25, min(92, adj_p))
    f_tw = max(0.45, min(2.7, adj_tw))
    return int(f_p), round(f_tw, 2), suggested_v

def perform_ai_engine(df, p_days, precision, trend_weight, v_comp):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    vol = df['Close'].pct_change().tail(20).std()
    sens = (int(precision) / 55)
    
    curr_p = float(last['Close'])
    open_p = float(last['Open'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100
    
    # 找回：準確性計算邏輯 (行數基準點)
    acc_base = 100 - (abs(curr_p - last['MA5']) / curr_p * 100)
    accuracy_val = min(99.4, max(68.0, acc_base + (precision / 15)))

    np.random.seed(42)
    sim_results = []
    trend = ((int(precision) - 55) / 1000) * float(trend_weight)
    for _ in range(1000):
        noise = np.random.normal(0, vol * v_comp, p_days)
        path = curr_p * np.cumprod(1 + trend + noise)
        sim_results.append(path)
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    all_first_day = [p[0] for p in sim_results]
    std_val = np.std(all_first_day)
    
    periods = {"5日短期": (last['MA5'], 0.8), "20日中期": (last['MA20'], 1.5), "60日長期": (last['MA60'], 2.2)}
    adv = {k: {"buy": m * (1 - vol * v_comp * f * sens), "sell": m * (1 + vol * v_comp * f * sens)} for k, (m, f) in periods.items()}
    
    score = (1 if curr_p > last['MA20'] else -1) + (1 if last['Hist'] > 0 else 0) + (1 if last['K'] < 25 else 0)
    status_map = {2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41")}
    res = status_map.get(score if score in status_map else -1, ("📉 偏空警戒", "#00FF41"))
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (res[0], "趨勢動能分析中", res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), accuracy_val)

# --- 4. 圖表與終端渲染 ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    final_p, final_tw, ai_suggested_v = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, v_comp)
    
    st.title(f"📊 {f_id} 實戰全能終端")
    st.caption(f"✨ AI 連動狀態：靈敏度 {final_p} | 趨勢增益 {final_tw} | 波動補償 {v_comp}")

    m_cols = st.columns(5)
    # 修正：將第三格放回 AI 預估準確性
    metrics = [
        ("當前價格", f"{curr_p:.2f}", "#FF3131" if change_pct >= 0 else "#00FF41"), 
        ("今日漲跌", f"{change_pct:.2f}%", "#FF3131" if change_pct >= 0 else "#00FF41"), 
        ("AI 預估準確性", f"{insight[6]:.1f}%", "#FFAC33"), 
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), 
        ("今日成交", f"{curr_v:,}", "#FFFF00")
    ]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04, subplot_titles=("價格與均線系統", "成交量分析", "MACD 能量柱", "KDJ 擺動指標"))
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name='K線走勢'), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測路徑', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    
    fig.update_layout(template="plotly_dark", height=880, xaxis_rangeslider_visible=False, margin=dict(r=180, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"<div class='ai-advice-box'><span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span><hr><p><b>診斷：</b>{insight[1]}</p><p>預估隔日收盤價：{insight[3]:.2f} (區間: {insight[5]:.2f} ~ {insight[4]:.2f})</p></div>", unsafe_allow_html=True)
    
    # 新增：寫入 predictions 分頁的按鈕
    if st.button("📝 記錄今日預測數據"):
        ws_p.append_row([datetime.now().strftime("%Y-%m-%d"), f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""])
        st.success("✅ 數據已同步至 Google Sheets 預測分頁！")

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s, ws_p = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings"), sh.worksheet("predictions")
    except: st.error("🚨 資料庫連線失敗"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp, api_ttl, tw_val, v_comp = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 1)), float(s_map.get('trend_weight', 1.0)), float(s_map.get('vol_comp', 1.5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 登入系統")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("確認登入", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
    else:
        with st.expander("⚙️ 終端設定面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                target = st.selectbox("自選清單", all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else ["2330"])
                if st.button("🗑️ 刪除目前選定股票"):
                    all_rows = ws_w.get_all_values()
                    for idx, row in enumerate(all_rows):
                        if row[0] == st.session_state.user and row[1] == target:
                            ws_w.delete_rows(idx + 1); st.rerun()
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    b1 = st.text_input("1. 權值標本-藍籌股基準", s_map.get('benchmark_1', '2330'))
                    b2 = st.text_input("2. 成長標本-高波動指標", s_map.get('benchmark_2', '2317'))
                    b3 = st.text_input("3. ETF 標本-市場資金流向", s_map.get('benchmark_3', '0050'))
                    new_p = st.slider("系統靈敏度", 0, 100, cp)
                    new_tw = st.number_input("AI 趨勢權重", 0.5, 3.0, tw_val)
                    new_ttl = st.number_input("API 快取 (1-10 分鐘)", 1, 10, api_ttl)
                    new_v = st.slider("波動補償係數", 0.5, 3.0, v_comp)
                    
                    if st.button("💾 同步觀察標本與學習參數"):
                        ws_s.update_cell(2, 2, str(new_p)); ws_s.update_cell(3, 2, str(new_ttl))
                        ws_s.update_cell(4, 2, b1); ws_s.update_cell(5, 2, b2); ws_s.update_cell(6, 2, b3)
                        ws_s.update_cell(7, 2, str(new_tw)); ws_s.update_cell(8, 2, str(new_v))
                        st.success("✅ 同步成功！"); st.rerun()
                if st.button("🚪 登出"): st.session_state.user = None; st.rerun()
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

if __name__ == "__main__": main()
