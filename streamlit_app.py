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

# --- 1. 配置與 UI 視覺深度強化 ---
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

# --- 2. 數據引擎 (優化 MultiIndex 處理與 2330 讀取) ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_tw_stock_data(symbol):
    s = str(symbol).strip().upper()
    # 確保代碼格式正確
    if not (s.endswith(".TW") or s.endswith(".TWO")):
        final_symbol = f"{s}.TW"
    else:
        final_symbol = s
    
    for attempt in range(3):
        try:
            # 使用 yf.download 並設定 auto_adjust=True 以簡化欄位
            df = yf.download(final_symbol, period="2y", interval="1d", auto_adjust=True, progress=False)
            
            # 關鍵修正：若 yf.download 返回 MultiIndex (新版本常見)，強制轉換為單層
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 若 2330.TW 失敗，嘗試 2330.TWO (雖然 2330 是上市，但此邏輯可增加整體穩定性)
            if (df is None or df.empty) and ".TW" in final_symbol:
                alt_symbol = final_symbol.replace(".TW", ".TWO")
                df = yf.download(alt_symbol, period="2y", interval="1d", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty: final_symbol = alt_symbol

            if df is not None and not df.empty:
                ticker_info = yf.Ticker(final_symbol)
                name = ticker_info.info.get('longName') or ticker_info.info.get('shortName') or final_symbol
                
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
                
                news = [n['title'] for n in ticker_info.news[:3]] if ticker_info.news else ["目前無即時新聞"]
                return df.dropna(), name, news, final_symbol
            time.sleep(1.5)
        except Exception as e:
            time.sleep(1.5); continue
    return None, symbol, [], symbol

# --- 3. AI 多週期深度分析邏輯 ---
def perform_ai_multi_period(df, precision):
    last = df.iloc[-1]
    bias = (int(precision) - 55) / 100
    volatility = df['Close'].pct_change().tail(20).std()
    
    periods = {
        "5日短期 (MA5基準)": {"ma": "MA5", "std": 1.6},
        "20日中期 (MA20基準)": {"ma": "MA20", "std": 2.4},
        "60日長期 (MA60基準)": {"ma": "MA60", "std": 3.8}
    }
    
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

# --- 4. 繪圖與儀表板渲染 ---
def render_dashboard(symbol, precision):
    df, name, news, final_id = fetch_tw_stock_data(symbol)
    if df is None: 
        st.error(f"❌ 無法讀取股票代碼 '{symbol}'。原因可能是：\n1. 網路連線至 Yahoo Finance 異常\n2. 該代碼目前無交易數據\n3. yfinance API 版本需要更新")
        return

    st.title(f"🇹🇼 {name} ({final_id})")
    
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

    # 4層技術指標圖
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.03)
    p_df = df.tail(70)
    
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5', line=dict(color='#FFFF00', width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20', line=dict(color='#00F5FF', width=1.5)), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Open'].iloc[i] > p_df['Close'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='成交量', marker_color=v_colors), 2, 1)
    
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD柱', marker_color=v_colors), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], name='MACD線', line=dict(color='#00F5FF')), 3, 1)
    
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K(加粗)', line=dict(color='#00FF41', width=3.5)), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D線', line=dict(color='#FFFF00', width=1.2)), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📰 市場即時新聞")
    for n in news:
        st.markdown(f"<div class='diag-box'>📢 {n}</div>", unsafe_allow_html=True)

# --- 5. 主程式 ---
def main():
    @st.cache_resource(ttl=1800)
    def get_session(): return {"user": None}
    
    vault = get_session()
    if 'user' not in st.session_state: st.session_state.user = vault["user"]

    try:
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings")
    except Exception as e:
        st.error(f"🚨 資料庫連線失敗: {e}"); return

    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    curr_prec = int(s_map.get('global_precision', 55))

    if st.session_state.user is None:
        st.title("🚀 StockAI 終端登入")
        u, p = st.text_input("帳號"), st.text_input("密碼", type="password")
        if st.button("確認進入"):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = vault["user"] = u; st.rerun()
            else: st.error("帳密錯誤")
    else:
        with st.expander("⚙️ 終端功能面板", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("切換股票", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("新增台股 (例: 2454)")
                if st.button("確認新增"):
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
            with m2:
                st.number_input("AI 預測天數 (最大值30)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    new_p = st.slider("靈敏度", 0, 100, curr_prec)
                    if st.button("💾 同步"): ws_s.update_cell(2, 2, str(new_p)); st.rerun()
                if st.button("🚪 登出"):
                    st.session_state.user = vault["user"] = None; st.rerun()
        
        render_dashboard(target, curr_prec)

if __name__ == "__main__": main()
