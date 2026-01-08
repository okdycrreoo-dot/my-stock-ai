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
    .gold-medal { color: #FFD700; font-weight: 900; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 雲端數據同步核心 ---
def get_db():
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        return {
            "users": sh.worksheet("users"), "watchlist": sh.worksheet("watchlist"),
            "settings": sh.worksheet("settings"), "ledger": sh.worksheet("ledger"),
            "rankings": sh.worksheet("rankings")
        }
    except: return None

db = get_db()

def sync_log(user, action, symbol, price, amount, balance):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    db['ledger'].append_row([user, now, action, symbol, price, amount, round(balance, 2)])

def check_monthly_reset(user):
    logs = pd.DataFrame(db['ledger'].get_all_records())
    user_logs = logs[logs['username'] == user]
    today = datetime.now()
    if not user_logs.empty:
        last_date = pd.to_datetime(user_logs.iloc[-1]['date'])
        if last_date.month != today.month:
            final_bal = float(user_logs.iloc[-1]['balance'])
            profit_pct = (final_bal - 1000000) / 1000000 * 100
            db['rankings'].append_row([last_date.strftime("%Y-%m"), user, f"{profit_pct:+.2f}%"])
            sync_log(user, "RESET", "SYSTEM", 0, 0, 1000000)
    elif user_logs.empty:
        sync_log(user, "INIT", "SYSTEM", 0, 0, 1000000)

# --- 3. 數據與 AI 引擎 (維持原始優化邏輯) ---
@st.cache_data(ttl=600)
def fetch_comprehensive_data(symbol):
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): s = f"{s}.TW"
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
        df['K'], df['D'] = rsv.ewm(com=2).mean(), rsv.ewm(com=2).mean().ewm(com=2).mean()
        return df.dropna(), s
    return None, s

def perform_ai_engine(df, p_days, precision):
    last, prev = df.iloc[-1], df.iloc[-2]
    vol = df['Close'].pct_change().tail(20).std()
    sens = (int(precision) / 55)
    curr_p, open_p, prev_c = float(last['Close']), float(last['Open']), float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100
    pred_prices = curr_p * np.cumprod(1 + (int(precision)-55)/1000 + np.random.normal(0, vol, p_days))
    
    periods = {"5日短期": (last['MA5'], 0.8), "20日中期": (last['MA20'], 1.5), "60日長期": (last['MA60'], 2.2)}
    adv = {k: {"buy": m * (1 - vol*f*sens), "sell": m * (1 + vol*f*sens)} for k, (m, f) in periods.items()}
    score = (1 if curr_p > last['MA20'] else -1) + (1 if last['Hist'] > 0 else 0) + (1 if last['K'] < 25 else 0)
    status_map = {2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41")}
    status, color = status_map.get(score, ("📉 偏空警戒", "#00FF41"))
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (status, color, pred_prices[0], pred_prices[0]*(1+vol), pred_prices[0]*(1-vol))

# --- 4. 渲染函數 ---
def render_terminal(symbol, p_days, precision):
    df, f_id = fetch_comprehensive_data(symbol)
    if df is None: return None
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, precision)
    
    st.title(f"📊 {f_id} 實戰全能終端")
    # 行情條
    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    m_cols = st.columns(5)
    metrics = [("當前價格", f"{curr_p:.2f}", c_p), ("今日漲跌", f"{change_pct:+.2f}%", c_p), ("今日開盤", f"{open_p:.2f}", "#FFF"), ("昨日收盤", f"{prev_c:.2f}", "#FFF"), ("成交量", f"{curr_v:,}", "#FFFF00")]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)
    
    # 圖表區 (維持專業佈局)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=2.5)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI預測', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # AI 展望
    st.markdown(f"""<div class='ai-advice-box'><span style='font-size:1.5rem; color:{insight[1]}; font-weight:900;'>{insight[0]}</span><hr><div style='background: #1C2128; padding: 12px; border-radius: 8px;'><p style='color:#00F5FF; font-weight:bold;'>🔮 明日 AI 展望：</p><p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估收盤：{insight[2]:.2f}</p></div></div>""", unsafe_allow_html=True)
    return curr_p

# --- 5. 主程式 ---
def main():
    if db is None: st.error("數據庫未連接"); return
    
    if 'user' not in st.session_state:
        st.title("🏆 StockAI 全員競技場")
        st.subheader("🥇 本月獲利排行榜 (最近 6 個月)")
        rank_df = pd.DataFrame(db['rankings'].get_all_records())
        if not rank_df.empty:
            pivot = rank_df.pivot(index='username', columns='month', values='profit_pct').fillna("-")
            # 微調：找出每個月的冠軍並加上金牌
            for col in pivot.columns:
                valid_vals = pivot[col][pivot[col] != "-"].str.rstrip('%').astype(float)
                if not valid_vals.empty:
                    winner = valid_vals.idxmax()
                    pivot.at[winner, col] = f"🥇 {pivot.at[winner, col]}"
            st.table(pivot.tail(6))
        
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("進入終端"):
            udf = pd.DataFrame(db['users'].get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; check_monthly_reset(u); st.rerun()
    else:
        user = st.session_state.user
        ledger_all = pd.DataFrame(db['ledger'].get_all_records())
        user_ledger = ledger_all[ledger_all['username'] == user]
        current_cash = float(user_ledger.iloc[-1]['balance']) if not user_ledger.empty else 1000000.0

        with st.expander("⚙️ 模擬交易與資產管理", expanded=False):
            c1, c2, c3 = st.columns([1, 1, 1.5])
            with c1:
                st.metric("本月可用資金", f"${current_cash:,.0f}")
                u_stocks = pd.DataFrame(db['watchlist'].get_all_records())
                u_stocks = u_stocks[u_stocks['username']==user]['stock_symbol'].tolist()
                target = st.selectbox("標的", u_stocks if u_stocks else ["2330"])
                if st.button("🚪 登出"): del st.session_state.user; st.rerun()
            with c2:
                st.write("📈 **交易操作**")
                if st.button("💰 買入 1 張 (費 0.15%)"):
                    d_temp, _ = fetch_comprehensive_data(target)
                    p_now = float(d_temp.iloc[-1]['Close'])
                    cost = p_now * 1000 * 1.0015 # 買入加 0.15%
                    if current_cash >= cost:
                        sync_log(user, "BUY", target, p_now, 1000, current_cash - cost)
                        st.rerun()
                if st.button("💸 賣出 1 張 (費+稅 0.45%)"):
                    # 檢查有無持倉 (簡易檢查 ledger)
                    buys = len(user_ledger[(user_ledger['symbol']==target) & (user_ledger['type']=="BUY")])
                    sells = len(user_ledger[(user_ledger['symbol']==target) & (user_ledger['type']=="SELL")])
                    if buys > sells:
                        d_temp, _ = fetch_comprehensive_data(target)
                        p_now = float(d_temp.iloc[-1]['Close'])
                        income = p_now * 1000 * (1 - 0.0045) # 賣出扣 0.45%
                        sync_log(user, "SELL", target, p_now, 1000, current_cash + income)
                        st.rerun()
                    else: st.warning("目前無此標的持倉")
            with c3:
                st.write("📊 **帳戶淨值走勢**")
                if not user_ledger.empty:
                    fig_c = go.Figure(go.Scatter(x=user_ledger['date'], y=user_ledger['balance'], fill='tozeroy', line=dict(color='#00F5FF')))
                    fig_c.update_layout(template="plotly_dark", height=180, margin=dict(t=0,b=0,l=0,r=0))
                    st.plotly_chart(fig_c, use_container_width=True)

        render_terminal(target, 7, 55)

if __name__ == "__main__": main()
