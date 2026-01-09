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

# --- 1. 配置與 UI 視覺 (完整還原 290 行版本，包含管理員自定義 CSS) ---
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
    .ai-advice-box { background-color: #161B22; border: 1px solid #FFAC33; border-radius: 12px; padding: 20px; margin-top: 15px; border-left: 10px solid #FFAC33; position: relative; }
    .price-buy { color: #FF3131; font-weight: 900; font-size: 1.3rem; }
    .price-sell { color: #00FF41; font-weight: 900; font-size: 1.3rem; }
    .realtime-val { font-size: 1.4rem; font-weight: 900; display: block; margin-top: 5px; }
    .label-text { color: #8899A6 !important; font-size: 0.8rem; letter-spacing: 1px; }
    .confidence-tag { position: absolute; top: 15px; right: 20px; color: #00F5FF; font-weight: 900; font-size: 0.9rem; border: 1px solid #00F5FF; padding: 2px 8px; border-radius: 15px; }
    
    /* 公告專用視覺 */
    .announcement-banner { 
        background: linear-gradient(90deg, #1C2128, #0E1117); 
        border: 2px solid #FF3131; 
        border-radius: 15px; 
        padding: 15px; 
        margin-bottom: 20px; 
        box-shadow: 0 0 15px rgba(255, 49, 49, 0.3); 
    }
    .scan-card { 
        background: #1C2128; 
        border: 1px solid #00F5FF; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center; 
    }
    
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (還原原始所有細節) ---
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

# --- 3. 背景自動對帳與命中率反饋 (完整還原) ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        today = datetime.now().strftime("%Y-%m-%d")
        
        for i, row in df_p.iterrows():
            if str(row['actual_close']) == "" and row['date'] != today:
                h = yf.download(row['symbol'], start=row['date'], end=(pd.to_datetime(row['date']) + timedelta(days=3)).strftime("%Y-%m-%d"), progress=False)
                if not h.empty:
                    act_close = float(h['Close'].iloc[0])
                    err_val = (act_close - float(row['pred_close'])) / float(row['pred_close'])
                    ws_p.update_cell(i + 2, 6, round(act_close, 2))
                    ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")

        if not any((r['date'] == today and r['symbol'] == f_id) for r in recs):
            new_row = [today, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""]
            ws_p.append_row(new_row)
        
        df_stock = df_p[(df_p['symbol'] == f_id) & (df_p['actual_close'] != "")].tail(10)
        if not df_stock.empty:
            hit = sum((df_stock['actual_close'] >= df_stock['range_low']) & (df_stock['actual_close'] <= df_stock['range_high']))
            return f"🎯 此股實戰命中率: {(hit/len(df_stock))*100:.1f}%"
        return "🎯 數據累積中"
    except:
        return "🎯 同步中"

# --- 4. AI 核心與連動引擎 (完整還原 290 行邏輯) ---
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
    
    np.random.seed(42)
    sim_results = []
    trend = ((int(precision) - 55) / 1000) * float(trend_weight)
    for _ in range(1000):
        noise = np.random.normal(0, vol * v_comp, p_days)
        path = curr_p * np.cumprod(1 + trend + noise)
        sim_results.append(path)
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    std_val = np.std([p[0] for p in sim_results])
    
    periods = {"5日短期": (last['MA5'], 0.8), "20日中期": (last['MA20'], 1.5), "60日長期": (last['MA60'], 2.2)}
    adv = {k: {"buy": m * (1 - vol * v_comp * f * sens), "sell": m * (1 + vol * v_comp * f * sens)} for k, (m, f) in periods.items()}
    
    score = (1 if curr_p > last['MA20'] else -1) + (1 if last['Hist'] > 0 else 0) + (1 if last['K'] < 25 else 0)
    status_map = {2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41")}
    res = status_map.get(score if score in status_map else -1, ("📉 偏空警戒", "#00FF41"))
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (res[0], "趨勢分析", res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5))

# --- [新功能]：全自動黃金交叉掃描系統 ---
def auto_market_scan_logic(ws_m, cp, tw_val, v_comp):
    today = datetime.now().strftime("%Y-%m-%d")
    records = ws_m.get_all_records()
    if any(r['date'] == today for r in records):
        return [r for r in records if r['date'] == today]
    
    # 種子標的清單
    seeds = ["2330", "2317", "2454", "2382", "2308", "2603", "2881", "0050"]
    picks = []
    for s in seeds:
        df_c, tid = fetch_comprehensive_data(s, 3600)
        if df_c is not None:
            _, _, _, _, _, _, _, insight = perform_ai_engine(df_c, 1, cp, tw_val, v_comp)
            # 判斷邏輯：MA5 即將金叉 MA20
            if df_c['MA5'].iloc[-1] < df_c['MA20'].iloc[-1] and insight[3] > df_c['MA20'].iloc[-1]:
                row = [today, tid, round(insight[3], 2), insight[0]]
                ws_m.append_row(row)
                picks.append({"date": today, "symbol": tid, "pred_close": row[2], "status": row[3]})
    return picks

# --- 5. 終端渲染 (完整還原所有視覺細節) ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: return

    final_p, final_tw, ai_suggested_v = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, v_comp)
    stock_accuracy = auto_sync_feedback(ws_p, f_id, insight)

    st.title(f"📊 {f_id} 實戰全能終端")
    st.caption(f"✨ AI 連動狀態：靈敏度 {final_p} | 趨勢增益 {final_tw} | 波動補償 {v_comp} (建議值: {ai_suggested_v})")

    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    m_cols = st.columns(5)
    metrics = [("當前價格", f"{curr_p:.2f}", c_p), ("今日漲跌", f"{change_pct:+.2f}%", c_p), ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), ("今日成交", f"{curr_v:,}", "#FFFF00")]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04)
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name='K線'), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量能', marker_color='#00F5FF'), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD', marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']]), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K', line=dict(color='#00F5FF')), 4, 1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""<div class='ai-advice-box'><div class='confidence-tag'>{stock_accuracy}</div><span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span><hr style='border:0.5px solid #444; margin:10px 0;'><p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估隔日收盤：{insight[3]:.2f}</p></div>""", unsafe_allow_html=True)

# --- 6. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        sh = gspread.authorize(Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s, ws_p, ws_m = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings"), sh.worksheet("predictions"), sh.worksheet("market_discovery")
    except: st.error("🚨 連線失敗"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp, api_ttl, tw_val, v_comp = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 1)), float(s_map.get('trend_weight', 1.0)), float(s_map.get('vol_comp', 1.5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 登入系統")
        u, p = st.text_input("帳號", key="login_u"), st.text_input("密碼", type="password", key="login_p")
        if st.button("確認登入"):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
    else:
        # --- 自動預測觸發 ---
        if st.session_state.user == "okdycrreoo":
            auto_market_scan_logic(ws_m, cp, tw_val, v_comp)

        # --- 全局公告欄 (向所有使用者公告) ---
        today_data = [r for r in ws_m.get_all_records() if r['date'] == datetime.now().strftime("%Y-%m-%d")]
        if today_data:
            st.markdown(f"<div class='announcement-banner'><h3 style='color:#FF3131; margin:0;'>📢 AI 今日黃金交叉預測公告</h3></div>", unsafe_allow_html=True)
            cols = st.columns(len(today_data) if len(today_data) < 6 else 6)
            for i, p in enumerate(today_data[:6]):
                with cols[i]: st.markdown(f"<div class='scan-card'><b style='color:#00F5FF;'>{p['symbol']}</b><br><small>預估: {p['pred_close']}</small><br><span style='color:#FF3131; font-size:0.7rem;'>{p['status']}</span></div>", unsafe_allow_html=True)

        with st.expander("⚙️ 終端設定面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("自選清單", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("➕ 快速新增")
                if st.button("新增股票"): ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    if st.button("💾 同步觀察標本"): ws_s.update_cell(2, 2, str(cp)); st.success("同步成功"); st.rerun()
                if st.button("🚪 登出"): st.session_state.user = None; st.rerun()
        
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

if __name__ == "__main__": main()
