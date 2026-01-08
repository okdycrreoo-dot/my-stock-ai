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

# --- 1. 配置與 UI 視覺 ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; color: #00F5FF !important; 
        border: 2px solid #00F5FF !important; border-radius: 12px !important;
        font-size: 1.2rem !important; font-weight: 900 !important;
    }
    div[data-baseweb="select"] > div { background-color: #1C2128 !important; color: #FFFFFF !important; border: 2px solid #00F5FF !important; }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stButton>button { 
        background-color: #00F5FF !important; color: #0E1117 !important; 
        border: none !important; border-radius: 12px; font-weight: 900 !important;
        height: 3.5rem !important; width: 100% !important;
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
@st.cache_data(ttl=600, show_spinner=False)
def fetch_comprehensive_data(symbol):
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): s = f"{s}.TW"
    for _ in range(3):
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
                df['MACD'] = e12 - e26
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                df['Hist'] = df['MACD'] - df['Signal']
                l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
                rsv = (df['Close'] - l9) / (h9 - l9 + 0.001) * 100
                df['K'] = rsv.ewm(com=2).mean()
                df['D'] = df['K'].ewm(com=2).mean()
                df['J'] = 3 * df['K'] - 2 * df['D']
                return df.dropna(), s
            time.sleep(1.5)
        except: time.sleep(1.5); continue
    return None, s

# --- 3. AI 核心與分析引擎 ---
def perform_ai_engine(df, p_days, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    vol = df['Close'].pct_change().tail(20).std()
    sens = (int(precision) / 55)
    
    curr_p, open_p, prev_c = float(last['Close']), float(last['Open']), float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100
    
    # AI 預測邏輯
    noise = np.random.normal(0, vol, p_days)
    trend = (int(precision) - 55) / 1000
    pred_prices = curr_p * np.cumprod(1 + trend + noise)
    
    # 預測明日細節 (明日為預測序列第1天)
    next_close = pred_prices[0]
    next_high = next_close * (1 + vol)
    next_low = next_close * (1 - vol)
    
    # 1. 實戰積極型參數 (調整因子 f 讓價格更靠近現價)
    periods = {
        "5日短期": (last['MA5'], 0.8),   # 積極捕捉短線回檔
        "20日中期": (last['MA20'], 1.5), # 強化中期佈局效率
        "60日長期": (last['MA60'], 2.2)  # 避免等待過深的回調
    }
    adv = {k: {"buy": m * (1 - vol*f*sens), "sell": m * (1 + vol*f*sens)} for k, (m, f) in periods.items()}
    
    # AI 評語邏輯
    score = 0
    reasons = []
    if curr_p > last['MA20']: score += 1; reasons.append("站上月線")
    else: score -= 1; reasons.append("破月線")
    if last['Hist'] > 0: score += 1; reasons.append("MACD多頭")
    if last['K'] < 25: score += 1; reasons.append("KDJ低位反彈")
    
    if score >= 2: status, color = "🚀 強力買入", "#FF3131"
    elif score == 1: status, color = "📈 偏多操作", "#FF7A7A"
    elif score == 0: status, color = "⚖️ 觀望中性", "#FFFF00"
    else: status, color = "📉 偏空警戒", "#00FF41"
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (status, " | ".join(reasons), color, next_close, next_high, next_low)

# --- 4. 圖表與終端渲染 ---
def render_terminal(symbol, p_days, precision):
    df, f_id = fetch_comprehensive_data(symbol)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, precision)
    st.title(f"📊 {f_id} 實戰版技術終端")

    # A. 橫向行情條
    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""
    m_cols = st.columns(5)
    metrics = [("當前價格", f"{curr_p:.2f}", c_p), ("今日漲跌", f"{sign}{change_pct:.2f}%", c_p), 
               ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), 
               ("今日成交", f"{curr_v:,}", "#FFFF00")]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]:
            st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    # B. AI 策略建議 (實戰積極型價格)
    st.write("") 
    s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]:
            st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    # C. 技術圖表 (加入 MA5, MA20, MA60)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(90)
    
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], 
                                 increasing_line_color='#FF3131', increasing_fillcolor='#FF3131',
                                 decreasing_line_color='#00FF41', decreasing_fillcolor='#00FF41', name='K線'), 1, 1)
    # 加入 2. MA5 與 MA60 線
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFFFF', width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA60'], name='MA60', line=dict(color='#FFAC33', width=1.8)), 1, 1)
    
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測線', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量', marker_color=v_colors), 2, 1)
    m_colors = ['#FF3131' if val >= 0 else '#00FF41' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD', marker_color=m_colors), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K', line=dict(color='#00F5FF', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J', line=dict(color='#FF00FF', width=1.2)), 4, 1)

    fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # D. 右下角 AI 綜合評語 (新增預測明日收盤與區間)
    st.markdown(f"""
    <div class='ai-advice-box'>
        <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
        <hr style='border:0.5px solid #444; margin:10px 0;'>
        <p style='font-size:1.1rem;'><b>分析依據：</b>{insight[1]}</p>
        <div style='background: #1C2128; padding: 10px; border-radius: 8px; border: 1px solid #444;'>
            <p style='margin:0; color:#00F5FF; font-weight:bold;'>🔮 AI 明日預測展望：</p>
            <p style='margin:5px 0 0 0; font-size:1.2rem; color:#FFAC33;'>預估收盤：{insight[3]:.2f}</p>
            <p style='margin:2px 0 0 0; font-size:0.95rem; color:#8899A6;'>預計波動區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
        </div>
        <p style='font-size:0.9rem; color:#8899A6; margin-top:10px;'>💡 建議：此價格為積極型參數設定。若明日股價進入預測區間下限，可考慮佈局。</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. 主程式 ---
def main():
    if 'user' not in st.session_state: st.session_state.user = None
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: st.error("🚨 資料庫連線失敗"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp = int(s_map.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🚀 StockAI 登入與註冊")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        with t1:
            u = st.text_input("帳號", key="login_u")
            p = st.text_input("密碼", type="password", key="login_p")
            if st.button("確認登入", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳號或密碼錯誤")
        with t2:
            nu = st.text_input("新帳號", key="reg_u")
            npw = st.text_input("新密碼", type="password", key="reg_p")
            if st.button("完成註冊", use_container_width=True):
                if nu and npw: ws_u.append_row([nu, npw]); st.success("註冊成功")
    else:
        with st.expander("⚙️ 終端管理面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("我的自選股", u_stocks if u_stocks else ["2330"])
                if st.button(f"🗑️ 刪除 {target}"):
                    vals = ws_w.get_all_values()
                    for i, r in enumerate(vals):
                        if i>0 and r[0]==st.session_state.user and r[1]==target:
                            ws_w.delete_rows(i+1); st.rerun()
                ns = st.text_input("➕ 新增股票代碼")
                if st.button("新增至清單"):
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                p_days = st.number_input("AI 預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("同步全域靈敏度", 0, 100, cp)
                    if st.button("💾 儲存並同步"):
                        ws_s.update_cell(2, 2, str(new_p)); st.rerun()
                if st.button("🚪 安全登出"): st.session_state.user = None; st.rerun()
        
        render_terminal(target, p_days, cp)

if __name__ == "__main__": main()
