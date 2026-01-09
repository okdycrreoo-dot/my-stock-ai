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

# --- 1. 配置與 UI 視覺 (完整還原原始代碼，不精簡) ---
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
    /* 新增：選股結果卡片樣式 */
    .scan-card { 
        background: #1C2128; 
        border: 1px solid #00F5FF; 
        padding: 15px; 
        border-radius: 12px; 
        text-align: center;
        margin-bottom: 10px;
    }
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (保留原始所有技術指標) ---
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

# --- 新增：自動同步、對帳與命中率計算 ---
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
    except: return "🎯 同步中"

# --- 3. AI 核心：蒙地卡羅預測引擎 ---
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
    res = status_map.get(score, ("📉 偏空警戒", "#00FF41"))
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (res[0], "趨勢確認", res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5))

# --- 新增：全市場掃描引擎邏輯 ---
def ai_market_discovery(cp, tw_val, v_comp):
    # 此處定義掃描清單（熱門標的）
    top_stocks = ["2330", "2317", "2454", "2382", "2308", "2603", "2609", "2409", "3481", "2881", "2882", "2357", "3034", "3037", "2379"]
    formatted = [f"{s}.TW" for s in top_stocks]
    st.info(f"🔍 AI 正掃描 {len(formatted)} 支核心標的之技術交叉動能...")
    
    try:
        # 第一階段：批量抓取
        data = yf.download(" ".join(formatted), period="1mo", interval="1d", group_by='ticker', progress=False)
        picks = []
        
        for t in formatted:
            d = data[t].dropna()
            if len(d) < 20: continue
            
            # 計算目前均線
            m5 = d['Close'].rolling(5).mean().iloc[-1]
            m20 = d['Close'].rolling(20).mean().iloc[-1]
            
            # 條件：MA5 尚未穿過 MA20，但距離小於 1.5%
            if m5 < m20 and (m20 - m5) / m20 < 0.015:
                # 第二階段：對這些標的進行深度 AI 預測
                df_c, _ = fetch_comprehensive_data(t, 3600)
                if df_c is not None:
                    _, _, _, _, _, _, _, insight = perform_ai_engine(df_c, 1, cp, tw_val, v_comp)
                    pred_next = insight[3]
                    # 明日預估 MA5
                    m5_next = (d['Close'].tail(4).sum() + pred_next) / 5
                    # 若預測明日會上穿
                    if m5_next > m20:
                        picks.append({"ticker": t, "pred": pred_next, "status": insight[0]})
        return picks
    except: return []

# --- 4. 圖表與終端渲染 (還原原始所有配置) ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, cp, tw_val, v_comp)
    stock_accuracy = auto_sync_feedback(ws_p, f_id, insight)

    st.title(f"📊 {f_id} 實戰全能終端")
    
    m_cols = st.columns(5)
    metrics = [("當前價格", f"{curr_p:.2f}", "#FF3131" if change_pct >= 0 else "#00FF41"), ("今日漲跌", f"{change_pct:+.2f}%", "#FF3131" if change_pct >= 0 else "#00FF41"), ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), ("今日成交", f"{curr_v:,}", "#FFFF00")]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04)
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測路徑', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color='#00F5FF'), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD', marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']]), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='KDJ-K', line=dict(color='#00F5FF')), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(r=20, t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
        <div class='ai-advice-box'>
            <div class='confidence-tag'>{stock_accuracy}</div>
            <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
            <hr style='border:0.5px solid #444; margin:10px 0;'>
            <p style='color:#00F5FF; font-weight:bold;'>🔮 AI 統一展望 (基準日: {df.index[-1].strftime('%Y/%m/%d')})：</p>
            <p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估下個交易日收盤：{insight[3]:.2f}</p>
            <p style='color:#8899A6;'>預估區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    if st.session_state.user and (time.time() - st.session_state.last_active > 600): st.session_state.user = None
    st.session_state.last_active = time.time()
    
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s, ws_p = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings"), sh.worksheet("predictions")
    except: st.error("🚨 資料庫連線失敗"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp = int(s_map.get('global_precision', 55))
    api_ttl = int(s_map.get('api_ttl_min', 1))
    tw_val = float(s_map.get('trend_weight', 1.0))
    v_comp = float(s_map.get('vol_comp', 1.5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 登入系統")
        u, p = st.text_input("帳號", key="login_u"), st.text_input("密碼", type="password", key="login_p")
        if st.button("確認登入", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
            else: st.error("驗證失敗")
    else:
        with st.expander("⚙️ 終端設定面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("自選清單", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("➕ 快速新增 (代碼)")
                if st.button("新增股票"): (ws_w.append_row([st.session_state.user, ns.upper()]), st.rerun()) if ns else None
                if st.button("🗑️ 刪除目前選定股票"):
                    all_rows = ws_w.get_all_values()
                    for idx, row in enumerate(all_rows):
                        if row[0] == st.session_state.user and row[1] == target:
                            ws_w.delete_rows(idx + 1); st.rerun()
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    # 這裡是您原本的管理員輸入項
                    b1 = st.text_input("基準藍籌 (2330)", s_map.get('benchmark_1', '2330'))
                    b2 = st.text_input("成長標本 (2317)", s_map.get('benchmark_2', '2317'))
                    new_p = st.slider("系統靈敏度", 0, 100, cp)
                    new_v = st.slider("波動補償", 0.5, 3.0, v_comp)
                    
                    # --- 嵌入：選股選單 ---
                    st.markdown("---")
                    st.markdown("#### 🎯 AI 全市場選股雷達")
                    if st.button("🚀 啟動明日黃金交叉預測"):
                        picks = ai_market_discovery(cp, tw_val, v_comp)
                        if picks:
                            st.write("💎 AI 預測明日交叉標的：")
                            p_cols = st.columns(3)
                            for idx, p in enumerate(picks):
                                with p_cols[idx % 3]:
                                    st.markdown(f"""
                                    <div class='scan-card'>
                                        <h4 style='color:#00F5FF;margin:0;'>{p['ticker']}</h4>
                                        <p style='font-size:0.8rem;margin:5px 0;'>預估: {p['pred']:.2f}</p>
                                        <p style='color:#FF3131;font-weight:900;'>{p['status']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else: st.info("今日無即將交叉標的。")
                    
                    if st.button("💾 同步觀察標本與學習參數"):
                        ws_s.update_cell(2, 2, str(new_p)); ws_s.update_cell(8, 2, str(new_v)); st.success("✅ 同步成功！"); st.rerun()
                if st.button("🚪 登出"): st.session_state.user = None; st.rerun()
        
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

if __name__ == "__main__": main()
