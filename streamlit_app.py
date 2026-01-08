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
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 18px; margin-bottom: 12px; border: 1px solid #30363D; }
    .ai-advice-box { background-color: #1C2128; border: 1px solid #FFAC33; border-radius: 12px; padding: 20px; margin-top: 15px; }
    .price-buy { color: #FF3131; font-weight: 900; font-size: 1.3rem; }
    .price-sell { color: #00FF41; font-weight: 900; font-size: 1.3rem; }
    .realtime-val { font-size: 1.4rem; font-weight: 900; }
    .label-text { color: #8899A6 !important; font-size: 0.85rem; }
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

# --- 3. AI 核心分析與建議生成 ---
def get_ai_insight(df, curr_p, change_pct):
    last = df.iloc[-1]
    k, d, j = last['K'], last['D'], last['J']
    macd_hist = last['Hist']
    ma20 = last['MA20']
    
    # 評級邏輯
    score = 0
    reasons = []
    
    # 1. 趨勢判定
    if curr_p > ma20:
        score += 1; reasons.append("股價位於站上月線，多頭結構穩健。")
    else:
        score -= 1; reasons.append("股價低於月線，短期空方佔優。")
        
    # 2. 動能判定 (MACD)
    if macd_hist > 0:
        score += 1; reasons.append("MACD紅柱擴張，上攻動能強勁。")
    else:
        score -= 1; reasons.append("MACD處於空方區域，謹防回檔。")
        
    # 3. 超買超賣 (KDJ)
    if k > 80: reasons.append("KDJ進入高檔超買區，不建議追高。")
    elif k < 20: score += 1; reasons.append("KDJ進入低檔超賣區，具備反彈潛力。")
    
    # 評級分類
    if score >= 2: status = "🚀 強力買入"; color = "#FF3131"
    elif score == 1: status = "📈 偏多操作"; color = "#FF7A7A"
    elif score == 0: status = "⚖️ 觀望中性"; color = "#FFFF00"
    else: status = "📉 偏空警戒"; color = "#00FF41"
    
    advice = " | ".join(reasons)
    return status, advice, color

def perform_ai_engine(df, p_days, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    vol = df['Close'].pct_change().tail(20).std()
    sens = (int(precision) / 55)
    
    curr_p, open_p, prev_c = float(last['Close']), float(last['Open']), float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100
    
    noise = np.random.normal(0, vol, p_days)
    trend = (int(precision) - 55) / 1000
    pred_prices = curr_p * np.cumprod(1 + trend + noise)
    
    periods = {"5日短期": (last['MA5'], 1.6), "20日中期": (last['MA20'], 2.6), "60日長期": (last['MA60'], 4.0)}
    adv = {k: {"buy": m * (1 - vol*f*sens), "sell": m * (1 + vol*f*sens)} for k, (m, f) in periods.items()}
    
    status, insight, s_color = get_ai_insight(df, curr_p, change_pct)
    
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (status, insight, s_color)

# --- 4. 圖表渲染 ---
def render_terminal(symbol, unit, p_days, precision):
    df, f_id = fetch_comprehensive_data(symbol)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight_data = perform_ai_engine(df, p_days, precision)
    st.title(f"📊 {f_id} 終端看板")

    # 配置配色
    color_code = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""

    # 面板 1: 行情即時數據庫
    m1, m2 = st.columns([1.5, 3])
    with m1:
        st.markdown(f"""
        <div class='diag-box'>
            <center><b>即時行情數據</b></center><hr style='border:0.5px solid #333'>
            <table style='width:100%'>
                <tr><td class='label-text'>當前價格</td><td class='realtime-val' style='color:{color_code}'>{curr_p:.2f}</td></tr>
                <tr><td class='label-text'>漲跌幅度</td><td class='realtime-val' style='color:{color_code}'>{sign}{change_pct:.2f}%</td></tr>
                <tr><td class='label-text'>今日開盤</td><td class='realtime-val'>{open_p:.2f}</td></tr>
                <tr><td class='label-text'>昨日收盤</td><td class='realtime-val'>{prev_c:.2f}</td></tr>
                <tr><td class='label-text'>今日成交</td><td class='realtime-val' style='color:#FFFF00'>{curr_v:,}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        cols = st.columns(3)
        for i, (label, p) in enumerate(ai_recs.items()):
            with cols[i]:
                st.markdown(f"""
                <div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #333'>
                買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>
                賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>
                """, unsafe_allow_html=True)

    # 繪圖區
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(90)
    
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], 
                                 increasing_line_color='#FF3131', increasing_fillcolor='#FF3131',
                                 decreasing_line_color='#00FF41', decreasing_fillcolor='#00FF41', name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測線', line=dict(color='#FFAC33', width=3, dash='dash')), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='量', marker_color=v_colors), 2, 1)
    m_colors = ['#FF3131' if val >= 0 else '#00FF41' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD柱', marker_color=m_colors), 3, 1)
    
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K', line=dict(color='#00F5FF', width=3)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J', line=dict(color='#FF00FF', width=1.2)), 4, 1)

    fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- 核心更新：圖表下方的 AI 綜合評語區塊 ---
    status, insight, s_color = insight_data
    st.markdown(f"""
    <div class='ai-advice-box'>
        <span style='font-size:1.4rem; color:{s_color}; font-weight:900;'>{status}</span>
        <hr style='border:0.5px solid #444; margin: 10px 0;'>
        <p style='color:#FFFFFF; font-size:1.1rem; line-height:1.6;'>
            <b>💡 AI 綜合診斷：</b>{insight}<br>
            <b>🎯 操作建議：</b>參考上方建議價區間進行分批布局。若股價跌破月線或KDJ高檔交叉向下，應適度減碼防範風險。
        </p>
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
        st.title("🚀 StockAI 登入系統")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        with t1:
            u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
            if st.button("確認登入", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("帳密錯誤")
        with t2:
            nu, npw = st.text_input("新帳號"), st.text_input("新密碼", type="password")
            if st.button("完成註冊", use_container_width=True):
                if nu: ws_u.append_row([nu, npw]); st.success("註冊成功")
    else:
        with st.expander("⚙️ 終端管理面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("我的選股", u_stocks if u_stocks else ["2330"])
                if st.button(f"🗑️ 刪除 {target}"):
                    vals = ws_w.get_all_values()
                    for i, r in enumerate(vals):
                        if i>0 and r[0]==st.session_state.user and r[1]==target:
                            ws_w.delete_rows(i+1); st.rerun()
                ns = st.text_input("➕ 新增代碼")
                if st.button("執行新增"):
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                p_days = st.number_input("AI 預測天數", 1, 30, 7)
                unit = st.selectbox("時間單位", ["日", "月", "年"])
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("同步全域靈敏度", 0, 100, cp)
                    if st.button("💾 更新同步"):
                        ws_s.update_cell(2, 2, str(new_p)); st.rerun()
                if st.button("🚪 安全登出"): st.session_state.user = None; st.rerun()
        
        render_terminal(target, unit, p_days, cp)

if __name__ == "__main__": main()
