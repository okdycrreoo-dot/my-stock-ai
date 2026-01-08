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

# --- 1. 配置與 UI 視覺深度強化 (要求: 全亮度、PWA 注入) ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { 
        color: #FFFFFF !important; font-weight: 800 !important; 
        text-shadow: 1px 1px 2px #000000;
    }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; color: #00F5FF !important; 
        border: 2px solid #00F5FF !important; border-radius: 12px !important;
        font-size: 1.1rem !important; font-weight: 900 !important;
    }
    div[data-baseweb="select"] > div { 
        background-color: #1C2128 !important; color: #FFFFFF !important; 
        border: 2px solid #00F5FF !important; 
    }
    input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stButton>button { 
        background-color: #00F5FF !important; color: #0E1117 !important; 
        border: 1px solid #FFFFFF !important; border-radius: 12px; 
        font-weight: 900 !important; height: 3.5rem !important; width: 100%;
    }
    .diag-box { 
        background-color: #161B22; border-left: 6px solid #00F5FF; 
        border-radius: 12px; padding: 18px; margin-bottom: 12px; border: 1px solid #30363D; 
    }
    .price-buy { font-size: 1.4rem; font-weight: 900; color: #00FF41; }
    .price-sell { font-size: 1.4rem; font-weight: 900; color: #FF3131; }
    button[data-testid="sidebar-button"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據引擎 (保留原始 3 次重試機制與台股自動補全) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_tw_stock_data(symbol):
    s = str(symbol).strip().upper()
    final_symbol = f"{s}.TW" if not (s.endswith(".TW") or s.endswith(".TWO")) else s
    
    for attempt in range(3):
        try:
            ticker = yf.Ticker(final_symbol)
            df = ticker.history(period="2y", interval="1d", auto_adjust=True)
            
            if (df is None or df.empty) and ".TW" in final_symbol:
                final_symbol = final_symbol.replace(".TW", ".TWO")
                ticker = yf.Ticker(final_symbol)
                df = ticker.history(period="2y", interval="1d", auto_adjust=True)

            if df is not None and not df.empty:
                info = ticker.info
                name = info.get('longName') or info.get('shortName') or final_symbol
                # 指標計算
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                # MACD
                e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
                df['MACD'] = e12 - e26
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                df['Hist'] = df['MACD'] - df['Signal']
                # KDJ
                l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
                rsv = (df['Close'] - l9) / (h9 - l9 + 0.001) * 100
                df['K'] = rsv.ewm(com=2).mean()
                df['D'] = df['K'].ewm(com=2).mean()
                df['J'] = 3 * df['K'] - 2 * df['D']
                
                news = [n['title'] for n in ticker.news[:3]] if ticker.news else ["目前無即時新聞"]
                return df.dropna(), name, news, final_symbol
            time.sleep(1.5)
        except:
            time.sleep(1.5); continue
    return None, symbol, [], symbol

# --- 3. AI 多週期深度分析邏輯 ---
def perform_ai_multi_period(df, precision):
    last = df.iloc[-1]
    bias = (int(precision) - 55) / 100
    volatility = df['Close'].pct_change().tail(20).std() # 取20日波動率
    
    periods = {
        "5日短期 (MA5基準)": {"ma": "MA5", "std": 1.6},
        "20日中期 (MA20基準)": {"ma": "MA20", "std": 2.4},
        "60日長期 (MA60基準)": {"ma": "MA60", "std": 3.8}
    }
    
    # 動能修正因子：考量 MACD 與 KDJ
    # 若 MACD 柱狀體增加且 K < 40，則代表超跌反彈機率高，略微提高買入價(積極布局)
    m_factor = 0.005 if last['Hist'] > df.iloc[-2]['Hist'] else -0.005
    k_factor = 0.01 if last['K'] < 30 else (-0.01 if last['K'] > 70 else 0)
    total_mod = 1 + m_factor + k_factor + bias

    res = {}
    for k, v in periods.items():
        base = last[v['ma']]
        res[k] = {
            "buy": base * (1 - (volatility * v['std'])) * total_mod,
            "sell": base * (1 + (volatility * v['std'])) * total_mod
        }
    return res

# --- 4. 繪圖與儀表板渲染 (4層圖表) ---
def render_dashboard(symbol, precision):
    df, name, news, final_id = fetch_tw_stock_data(symbol)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗，請檢查代碼。"); return

    st.title(f"🇹🇼 {name} ({final_id})")
    
    # 週期建議價 (5/20/60)
    adv = perform_ai_multi_period(df, precision)
    cols = st.columns(3)
    for i, (period, prices) in enumerate(adv.items()):
        with cols[i]:
            st.markdown(f"""
            <div class='diag-box'>
                <center><b>{period}</b></center><hr style='border:0.5px solid #30363D'>
                買入參考: <span class='price-buy'>{prices['buy']:.2f}</span><br>
                賣出參考: <span class='price-sell'>{prices['sell']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # 圖表層 (要求 3: KDJ 綠線加粗)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(70)
    
    # 1. K線 + 均線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1)), 1, 1)
    
    # 2. 成交量
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    
    # 3. MACD
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD柱', marker_color=v_colors), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD線', line=dict(color='#00F5FF')), 3, 1)
    
    # 4. KDJ (要求 3: 綠線加粗)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(加粗)', line=dict(color='#00FF41', width=3.5)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J線', line=dict(color='#FF00FF', width=1.2)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 新聞摘要
    st.markdown("### 📰 市場即時新聞與趨勢")
    for n in news:
        st.markdown(f"<div class='diag-box'>📢 {n}</div>", unsafe_allow_html=True)

# --- 5. 主程式 (長效登入與 GSheet 管理) ---
def main():
    @st.cache_resource(ttl=1800) # 要求 5: 30分鐘持久
    def get_session(): return {"user": None}
    
    vault = get_session()
    if 'user' not in st.session_state: st.session_state.user = vault["user"]

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except: st.error("🚨 雲端數據庫連線異常"); return

    # 獲取管理參數
    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    curr_prec = int(s_map.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🚀 StockAI 終端登入")
        t1, t2 = st.tabs(["🔑 登入", "📝 註冊"])
        with t1:
            u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
            if st.button("確認進入"):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = vault["user"] = u; st.rerun()
                else: st.error("帳密錯誤")
        with t2:
            nu, npw = st.text_input("新帳號"), st.text_input("新密碼", type="password")
            if st.button("註冊"):
                if nu: ws_u.append_row([nu, npw]); st.success("註冊成功")
    else:
        with st.expander("⚙️ 功能管理面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                st.subheader("📋 我的清單")
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist() if not all_w.empty else []
                target = st.selectbox("切換股票", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("新增台股代碼 (例: 2454)")
                if st.button("確認新增"):
                    if ns and ns.upper() not in u_stocks:
                        ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                st.subheader("🛠️ 系統參數")
                # 要求 2: 文字標註
                st.number_input("AI 預測天數 (最大值30)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("全域靈敏度", 0, 100, curr_prec)
                    if st.button("💾 同步雲端"):
                        ws_s.update_cell(2, 2, str(new_p)); st.rerun()
                if st.button("🚪 安全登出"):
                    st.session_state.user = vault["user"] = None; st.rerun()
        
        render_dashboard(target, curr_prec)

if __name__ == "__main__": main()
