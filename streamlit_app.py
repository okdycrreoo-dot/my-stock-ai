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

# --- 1. 配置與 UI 視覺 (完整保留 290 行版本的所有 CSS，絕不精簡) ---
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
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
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
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                df['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))
                
                tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(14).mean()
                return df.dropna(), s
            time.sleep(1.5)
        except: 
            time.sleep(1.5)
            continue
    return None, s

# --- 3. 背景自動對帳與命中率反饋 (雙重防禦版) ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        today = datetime.now().strftime("%Y-%m-%d")
        is_weekend = datetime.now().weekday() >= 5

        for i, row in df_p.iterrows():
            if not is_weekend and str(row['actual_close']) == "" and row['date'] != today:
                h = yf.download(row['symbol'], start=row['date'], end=(pd.to_datetime(row['date']) + timedelta(days=3)).strftime("%Y-%m-%d"), progress=False)
                if not h.empty:
                    act_close = float(h['Close'].iloc[0])
                    err_val = (act_close - float(row['pred_close'])) / float(row['pred_close'])
                    ws_p.update_cell(i + 2, 6, round(act_close, 2))
                    ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")

        if not is_weekend and not any((r['date'] == today and r['symbol'] == f_id) for r in recs):
            new_row = [today, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""]
            ws_p.append_row(new_row)
        
        df_stock = df_p[(df_p['symbol'] == f_id) & (df_p['actual_close'] != "")].copy()
        if not df_stock.empty:
            df_stock = df_stock.loc[df_stock['actual_close'].shift() != df_stock['actual_close']]
            df_recent = df_stock.tail(10)
            hit = sum((df_recent['actual_close'] >= df_recent['range_low']) & (df_recent['actual_close'] <= df_recent['range_high']))
            return f"🎯 此股實戰命中率: {(hit/len(df_recent))*100:.1f}%"
        return "🎯 數據累積中"
    except:
        return "🎯 同步中"

# --- 4. AI 核心：深度微調連動引擎 (已升級：誤差反饋修正) ---
# ... (auto_fine_tune_engine 保持不變) ...

def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol, error_offset=0):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    sens = (int(precision) / 55)
    curr_p = float(last['Close'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100

    # --- 集中度偏移算法 ---
    v_avg20 = df['Volume'].tail(20).mean() 
    vol_ratio = curr_v / (v_avg20 + 0.1)
    if change_pct > 0.5 and vol_ratio > 1.2:
        chip_mom = (change_pct / 100) * vol_ratio * 1.5 
    elif change_pct < 0 and vol_ratio < 0.7:
        chip_mom = abs(change_pct / 100) * 0.2 
    elif change_pct < -1.5 and vol_ratio > 1.5:
        chip_mom = (change_pct / 100) * vol_ratio * 1.2
    else:
        chip_mom = (change_pct / 100)

    # 2. RSI 群體背離分析
    rsi_p = [5, 10, 15, 20, 25, 30]
    div_scores = []
    for p in rsi_p:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rsi_now = 100 - (100 / (1 + (gain / (loss + 1e-5)))).iloc[-1]
        rsi_prev = 100 - (100 / (1 + (gain / (loss + 1e-5)))).iloc[-2]
        d = -1 if (curr_p > prev_c and rsi_now < rsi_prev) else (1 if (curr_p < prev_c and rsi_now > rsi_prev) else 0)
        div_scores.append(d)
    rsi_div = sum(div_scores) / len(div_scores)
    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    # 3. 蒙特卡羅路徑模擬 (注入 error_offset 反饋補償)
    np.random.seed(42)
    sim_results = []
    # 這裡將過去的預測誤差 (error_offset) 納入 Drift 計算
    base_drift = ((int(precision) - 55) / 1000) * float(trend_weight) + (rsi_div * 0.002) + (chip_mom * 0.15) - (error_offset * 0.1)
    
    for _ in range(1000):
        noise = np.random.normal(0, f_vol * v_comp * vol_contract, p_days)
        path = [curr_p]
        for i in range(p_days):
            reversion_pull = bias * 0.08
            next_p = path[-1] * (1 + base_drift - reversion_pull + noise[i])
            path.append(next_p)
        sim_results.append(path[1:])
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    std_val = np.std([p[0] for p in sim_results])
    
    # 4. 6-MA 綜合診斷
    ma_check_list = [5, 10, 15, 20, 25, 30]
    above_ma_count = sum(1 for p in ma_check_list if curr_p > df['Close'].rolling(p).mean().iloc[-1])
    score = 0
    reasons = []
    if above_ma_count >= 5: score += 2; reasons.append(f"均線多頭({above_ma_count}/6)")
    elif above_ma_count <= 1: score -= 2; reasons.append(f"均線空頭({6-above_ma_count}/6)")
    if change_pct > 1.2 and vol_ratio > 1.3: score += 1; reasons.append("法人級放量攻擊")
    elif change_pct < -1.2 and vol_ratio > 1.3: score -= 1; reasons.append("法人級拋售壓力")
    if last['Hist'] > 0: score += 1; reasons.append("MACD多頭")
    if rsi_div >= 0.3: score += 1; reasons.append("RSI底背離")
    
    status_map = {3: ("🚀 強力買入", "#FF3131"), 2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41"), -2: ("📉 偏空警戒", "#00FF41")}
    res = status_map.get(max(-2, min(3, score)), ("⚖️ 觀望中性", "#FFFF00"))
    
    # 5. 實戰建議價格與乖離匯總
    periods = {"5日極短線建議": (df['Close'].rolling(5).mean().iloc[-1], 0.8), "10日短線建議": (df['Close'].rolling(10).mean().iloc[-1], 1.1), "20日波段建議": (last['MA20'], 1.5)}
    adv = {k: {"buy": m * (1 - f_vol * v_comp * f * sens), "sell": m * (1 + f_vol * v_comp * f * sens)} for k, (m, f) in periods.items()}
    b_sum = {p: (curr_p - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5) for p in [5, 10, 20, 30]}
    
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), b_sum)
    
    # 5. 實戰建議價格與乖離匯總
    periods = {"5日極短線買賣建議": (df['Close'].rolling(5).mean().iloc[-1], 0.8), "10日短線買賣建議": (df['Close'].rolling(10).mean().iloc[-1], 1.1), "20日波段買賣建議": (last['MA20'], 1.5)}
    adv = {k: {"buy": m * (1 - f_vol * v_comp * f * sens), "sell": m * (1 + f_vol * v_comp * f * sens)} for k, (m, f) in periods.items()}
    b_sum = {p: (curr_p - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5) for p in [5, 10, 20, 30]}
    
    # 正確對齊 8 個回傳項目
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), b_sum)
# --- 5. 圖表與終端渲染 (已整合 AI 誤差自我修正) ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗"); return

    # 1. 獲取過去誤差數據 (機器學習閉環的核心)
    recs = ws_p.get_all_records()
    df_p = pd.DataFrame(recs)
    df_stock = df_p[(df_p['symbol'] == f_id) & (df_p['actual_close'] != "")].tail(10)
    
    # 計算平均誤差百分比 (若預測太高，error_offset 為正；若太低，為負)
    try:
        def clean_pct(x): return float(str(x).replace('%','')) / 100 if x else 0
        error_offset = df_stock['error_pct'].apply(clean_pct).mean() if not df_stock.empty else 0
    except:
        error_offset = 0

    # 2. 執行 AI 引擎 (傳入 error_offset)
    final_p, final_tw, ai_v, _, bias, f_vol = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, ai_v, bias, f_vol, error_offset)
    
    stock_accuracy = auto_sync_feedback(ws_p, f_id, insight)

    # ... (其餘 UI 渲染、Metrics、Plotly 圖表代碼保持 290 行基準版不變) ...
    # 注意：請確保繪圖部分使用上面算出的新 pred_line
    st.title(f"📊 {f_id} 台股AI自主預測系統")
    if abs(error_offset) > 0.02:
        st.caption(f"🤖 AI 自我修正中：偵測到近期預測偏{'高' if error_offset > 0 else '低'}，已自動補償 {abs(error_offset):.2%} 的預測偏移。")
    
    # (此處接續您原本的 Metrics 和圖表渲染代碼...)
# --- 6. 主程式 (AI 全自主觀察室版本) ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    if st.session_state.user and (time.time() - st.session_state.last_active > 3600): st.session_state.user = None
    st.session_state.last_active = time.time()
    
    @st.cache_resource(ttl=30)
    def get_gsheets_connection():
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        return {"users": sh.worksheet("users"), "watchlist": sh.worksheet("watchlist"), "settings": sh.worksheet("settings"), "predictions": sh.worksheet("predictions")}

    try:
        sheets = get_gsheets_connection()
        ws_u, ws_w, ws_s, ws_p = sheets["users"], sheets["watchlist"], sheets["settings"], sheets["predictions"]
        s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
        cp_base, api_ttl, tw_base, v_base = 55, int(s_map.get('api_ttl_min', 1)), 1.0, 1.5
    except:
        st.error("🚨 資料庫連線失敗"); return

    if st.session_state.user is None:
        st.title("🚀 StockAI 台股預測系統")
        u = st.text_input("帳號", key="login_u")
        p = st.text_input("密碼", type="password", key="login_p")
        if st.button("登入"):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf.empty and not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
    else:
        all_w = pd.DataFrame(ws_w.get_all_records())
        u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
        
        with st.sidebar:
            st.header("🛡️ AI 觀察儀表板")
            target = st.selectbox("監測目標", u_stocks if u_stocks else ["2330"])
            p_days = st.slider("預測深度", 1, 30, 7)
            
            # AI 自主計算當前最優參數
            temp_df, _ = fetch_comprehensive_data(target, api_ttl*60)
            if temp_df is not None:
                ai_p, ai_tw, ai_v, _, _, _ = auto_fine_tune_engine(temp_df, cp_base, tw_base, v_base)
                st.markdown("---")
                st.metric("核心靈敏度", f"{ai_p}%", "AI 自適應")
                st.metric("趨勢權重", f"{ai_tw}x", "動態優化")
                st.metric("波動補償", f"{ai_v}v", "環境感知")
                
                if st.session_state.user == "okdycrreoo":
                    with st.expander("📝 觀察清單管理"):
                        ns = st.text_input("➕ 新增代碼")
                        if st.button("執行新增") and ns:
                            ws_w.append_row([st.session_state.user, ns.upper().strip()]); st.rerun()
                        if st.button("🗑️ 移除此股"):
                            all_rows = ws_w.get_all_values()
                            for i, r in reversed(list(enumerate(all_rows))):
                                if r[0] == st.session_state.user and r[1] == target:
                                    ws_w.delete_rows(i + 1); break
                            st.rerun()
            if st.button("🚪 登出"): st.session_state.user = None; st.rerun()

        # 這裡直接使用 AI 計算出的 ai_p, ai_tw, ai_v，達成全自主判斷
        render_terminal(target, p_days, ai_p, ai_tw, api_ttl, ai_v, ws_p)

if __name__ == "__main__":
    main()







