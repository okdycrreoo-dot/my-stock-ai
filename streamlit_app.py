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

# --- 1. 配置與 UI 視覺 (文字顏色修正) ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    
    /* 強制輸入框文字為黑色，背景為白色，確保看得到輸入代碼 */
    input { 
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important; 
        font-weight: 600 !important; 
    }
    div[data-baseweb="input"] { 
        background-color: #FFFFFF !important; 
        border-radius: 8px; 
    }
    
    /* 強制下拉選單選中文字為黑色 */
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

# --- 3. AI 核心與分析引擎 (還原 1,000 次模擬邏輯) ---
def perform_ai_engine(df, p_days, precision, trend_weight):
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
        noise = np.random.normal(0, vol, p_days)
        path = curr_p * np.cumprod(1 + trend + noise)
        sim_results.append(path)
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    all_first_day = [p[0] for p in sim_results]
    std_val = np.std(all_first_day)
    next_high = next_close + (std_val * 1.5)
    next_low = next_close - (std_val * 1.5)
    
    periods = {
        "5日短期": (last['MA5'], 0.8), 
        "20日中期": (last['MA20'], 1.5), 
        "60日長期": (last['MA60'], 2.2)
    }
    adv = {k: {"buy": m * (1 - vol*f*sens), "sell": m * (1 + vol*f*sens)} for k, (m, f) in periods.items()}
    
    score = 0
    reasons = []
    if curr_p > last['MA20']: 
        score += 1
        reasons.append("站上月線")
    else: 
        score -= 1
        reasons.append("破月線")
    if last['Hist'] > 0: 
        score += 1
        reasons.append("MACD多頭")
    if last['K'] < 25: 
        score += 1
        reasons.append("KDJ低位反彈")
    
    status_map = {
        2: ("🚀 強力買入", "#FF3131"), 
        1: ("📈 偏多操作", "#FF7A7A"), 
        0: ("⚖️ 觀望中性", "#FFFF00"), 
        -1: ("📉 偏空警戒", "#00FF41")
    }
    res = status_map.get(score if score in status_map else -1, ("📉 偏空警戒", "#00FF41"))
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_high, next_low)

# --- 4. 圖表與終端渲染 (分層說明標記) ---
def render_terminal(symbol, p_days, precision, trend_weight, ttl_min):
    df, f_id = fetch_comprehensive_data(symbol, ttl_min * 60)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗")
        return

    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, precision, trend_weight)
    st.title(f"📊 {f_id} 實戰全能終端")

    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""
    m_cols = st.columns(5)
    metrics = [
        ("當前價格", f"{curr_p:.2f}", c_p), 
        ("今日漲跌", f"{sign}{change_pct:.2f}%", c_p), 
        ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), 
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), 
        ("今日成交", f"{curr_v:,}", "#FFFF00")
    ]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: 
            st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    st.write("") 
    s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: 
            st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03, subplot_titles=("價格與均線", "成交量", "MACD", "KDJ"))
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name='K線', showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=2), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA60'], name='MA60', line=dict(color='#FFAC33', width=2), showlegend=False), 1, 1)
    
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI預測', line=dict(color='#FF3131', width=3, dash='dash'), showlegend=False), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量能', marker_color=v_colors, showlegend=False), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD', marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']], showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K值', line=dict(color='#00F5FF'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D值', line=dict(color='#FFFF00'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J值', line=dict(color='#E066FF'), showlegend=False), 4, 1)

    annos = [("均線/AI預測", 0.92, "#FFFFFF"), ("成交量能", 0.58, "#8899A6"), ("MACD力道", 0.38, "#FF7A7A"), ("KDJ (藍K/黃D/紫J)", 0.12, "#00F5FF")]
    for txt, y_p, clr in annos:
        fig.add_annotation(xref="paper", yref="paper", x=1.01, y=y_p, text=f"<b>{txt}</b>", showarrow=False, align="left", xanchor="left", font=dict(size=13, color=clr))

    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(r=160))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class='ai-advice-box'>
        <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
        <hr style='border:0.5px solid #444; margin:10px 0;'>
        <p><b>診斷：</b>{insight[1]}</p>
        <div style='background: #1C2128; padding: 12px; border-radius: 8px;'>
            <p style='color:#00F5FF; font-weight:bold;'>🔮 AI 統一展望 (1,000次模擬)：</p>
            <p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估隔日收盤價：{insight[3]:.2f}</p>
            <p style='color:#8899A6;'>預估隔日浮動區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    if st.session_state.user and time.time() - st.session_state.last_active > 600: st.session_state.user = None
    st.session_state.last_active = time.time()
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: 
        st.error("🚨 資料庫連線失敗"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    try: cp, api_ttl, tw_val = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 1)), float(s_map.get('trend_weight', 1.0))
    except: cp, api_ttl, tw_val = 55, 1, 1.0

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
                if st.button("新增股票"):
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    b1 = st.text_input("1. 權值標本(藍籌股基準：校準市場地基。建議: 2330.TW)", s_map.get('benchmark_1', '2330'))
                    b2 = st.text_input("2. 成長標本(高波動指標：識別噴發動能。建議: 2317.TW)", s_map.get('benchmark_2', '2317'))
                    b3 = st.text_input("3. ETF標本(資金流向：過濾個股雜訊。建議: 0050.TW)", s_map.get('benchmark_3', '0050'))
                    new_p, new_tw, new_ttl = st.slider("系統靈敏度", 0, 100, cp), st.number_input("AI 趨勢權重(預測增益。建議: 1.0~1.5)", 0.5, 3.0, tw_val), st.number_input("API 快取(分鐘)", 1, 10, api_ttl)
                    if st.button("💾 同步觀察標本與學習參數"):
                        ws_s.update_cell(2, 2, str(new_p)); ws_s.update_cell(3, 2, str(new_ttl)); ws_s.update_cell(4, 2, b1); ws_s.update_cell(5, 2, b2); ws_s.update_cell(6, 2, b3); ws_s.update_cell(7, 2, str(new_tw))
                        st.success("✅ 同步成功！"); st.rerun()
                if st.button("🚪 登出"): st.session_state.user = None; st.rerun()
        render_terminal(target, p_days, cp, tw_val, api_ttl)

if __name__ == "__main__":
    main()
