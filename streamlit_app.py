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
            df = yf.download(s, period="2y", interval="1d", auto_adjust=False, progress=False)
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
        is_weekend = datetime.now().weekday() >= 5  # 判定是否為週六或週日

        # 1. 【計算防禦】對帳部分：僅在非假日執行 yfinance 抓取
        for i, row in df_p.iterrows():
            if not is_weekend and str(row['actual_close']) == "" and row['date'] != today:
                h = yf.download(row['symbol'], start=row['date'], end=(pd.to_datetime(row['date']) + timedelta(days=3)).strftime("%Y-%m-%d"), progress=False)
                if not h.empty:
                    act_close = float(h['Close'].iloc[0])
                    err_val = (act_close - float(row['pred_close'])) / float(row['pred_close'])
                    ws_p.update_cell(i + 2, 6, round(act_close, 2))
                    ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")

        # 2. 【寫入防禦】僅在「非假日」且「今日尚未紀錄」時才新增行
        if not is_weekend and not any((r['date'] == today and r['symbol'] == f_id) for r in recs):
            new_row = [today, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""]
            ws_p.append_row(new_row)
        
        # 3. 【計算防禦】命中率計算：強制排除價格連續重複的紀錄
        df_stock = df_p[(df_p['symbol'] == f_id) & (df_p['actual_close'] != "")].copy()
        
        if not df_stock.empty:
            # 核心防禦：若價格連續相同則視為無效交易日，不計入分母
            df_stock = df_stock.loc[df_stock['actual_close'].shift() != df_stock['actual_close']]
            
            df_recent = df_stock.tail(10)
            hit = sum((df_recent['actual_close'] >= df_recent['range_low']) & 
                      (df_recent['actual_close'] <= df_recent['range_high']))
            
            return f"🎯 此股實戰命中率: {(hit/len(df_recent))*100:.1f}%"
        
        return "🎯 數據累積中"
    except:
        return "🎯 同步中"
        
# --- 4. AI 核心：深度微調連動引擎 (升級：均值回歸/量價加權/波動融合) ---
def auto_fine_tune_engine(df, base_p, base_tw, v_comp):
    rets = df['Close'].pct_change().dropna()
    
    # [進化一：波動融合 Volatility Fusion]
    v_5 = rets.tail(5).std()
    v_20 = rets.tail(20).std()
    v_60 = rets.tail(60).std()
    f_vol = (v_5 * 0.5) + (v_20 * 0.3) + (v_60 * 0.2)
    
    # [進化二：量價加權 Volume Weighting]
    v_curr = df['Volume'].iloc[-1]
    v_avg5 = df['Volume'].tail(5).mean()
    vol_spike = v_curr / (v_avg5 + 0.1)
    # 根據成交量異動調整趨勢權重
    f_tw = max(0.5, min(2.5, 1.0 + (rets.tail(5).mean() * 15 * min(1.5, vol_spike))))
    
    # [進化三：均值回歸 Bias Correction]
    price_now = float(df['Close'].iloc[-1])
    ma20_val = df['MA20'].iloc[-1]
    bias_val = (price_now - ma20_val) / (ma20_val + 0.1)
    
    # AI 全面優化配置
    f_p = 45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60
    high_low_range = (df['High'] - df['Low']).tail(5).mean() / price_now
    f_v = 1.3 if high_low_range > 0.035 else 2.1 if high_low_range < 0.015 else 1.7
    
    # AI 自動匹配推薦標本
    benchmarks = ("2330", "2382", "00878") if f_vol > 0.02 else ("2317", "2454", "0050")
    
    return int(f_p), round(f_tw, 2), f_v, benchmarks, bias_val, f_vol

def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # 使用融合波動率進行模擬
    sens = (int(precision) / 55)
    
    curr_p = float(last['Close'])
    open_p = float(last['Open'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100

    rsi_now, rsi_prev = last['RSI'], prev['RSI']
    rsi_div = -1 if (curr_p > prev_c and rsi_now < rsi_prev) else (1 if (curr_p < prev_c and rsi_now > rsi_prev) else 0)
    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    np.random.seed(42)
    sim_results = []
    # 核心趨勢包含：設定趨勢 + 均值回歸拉力 (bias * 0.08)
    base_drift = ((int(precision) - 55) / 1000) * float(trend_weight) + (rsi_div * 0.002)
    
    for _ in range(1000):
        noise = np.random.normal(0, f_vol * v_comp * vol_contract, p_days)
        # 加入均值回歸邏輯的路徑演化
        path = [curr_p]
        for i in range(p_days):
            reversion_pull = bias * 0.08
            next_p = path[-1] * (1 + base_drift - reversion_pull + noise[i])
            path.append(next_p)
        sim_results.append(path[1:])
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    all_first_day = [p[0] for p in sim_results]
    std_val = np.std(all_first_day)
    
    periods = {"5日短期": (last['MA5'], 0.8), "20日中期": (last['MA20'], 1.5), "60日長期": (last['MA60'], 2.2)}
    adv = {k: {"buy": m * (1 - f_vol * v_comp * f * sens), "sell": m * (1 + f_vol * v_comp * f * sens)} for k, (m, f) in periods.items()}
    
    score = 0
    reasons = []
    if curr_p > last['MA20']: score += 1; reasons.append("站上月線")
    else: score -= 1; reasons.append("破月線")
    if last['Hist'] > 0: score += 1; reasons.append("MACD多頭")
    if last['K'] < 25: score += 1; reasons.append("KDJ低位反彈")
    if rsi_div == 1: score += 1; reasons.append("RSI底背離(籌碼回補)")
    elif rsi_div == -1: score -= 1; reasons.append("RSI頂背離(主力派發)")
    if vol_contract < 0.8: reasons.append("ATR高度收縮(即將變盤)")
    status_map = {2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41")}
    res = status_map.get(score if score in status_map else -1, ("📉 偏空警戒", "#00FF41"))
    return pred_prices, adv, curr_p, open_p, prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5))
# --- 5. 圖表與終端渲染 ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: st.error(f"❌ 讀取 {symbol} 失敗"); return

    # 渲染時調用升級後的引擎
    final_p, final_tw, ai_v, _, bias, f_vol = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, ai_v, bias, f_vol)
    
    stock_accuracy = auto_sync_feedback(ws_p, f_id, insight)

    st.title(f"📊 {f_id} 實戰全能終端")
    st.caption(f"✨ AI 三大腦升級：均值回歸控管 | 量價加權權重 | 波動融合引擎 (已根據證交所市價同步)")

    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""
    m_cols = st.columns(5)
    metrics = [("當前價格", f"{curr_p:.2f}", c_p), ("今日漲跌", f"{sign}{change_pct:.2f}%", c_p), ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), ("今日成交 (張)", f"{int(curr_v/1000):,}", "#FFFF00")]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04, subplot_titles=("價格與均線系統 (含 AI 預測)", "成交量分析 (張)", "MACD 能量柱", "KDJ 擺動指標"))
    p_df = df.tail(90)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name='K線走勢', legendgroup="1"), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name='MA5 均線', line=dict(color='#FFFF00', width=2), legendgroup="1"), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], name='MA20 均線', line=dict(color='#00F5FF', width=1.5), legendgroup="1"), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA60'], name='MA60 均線', line=dict(color='#FFAC33', width=2), legendgroup="1"), 1, 1)
    
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 預測路徑', line=dict(color='#FF3131', width=3, dash='dash'), legendgroup="1"), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume']/1000, name='成交量 (張)', marker_color=v_colors, legendgroup="2"), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD 力道', marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']], legendgroup="3"), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K值 (藍)', line=dict(color='#00F5FF'), legendgroup="4"), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D值 (黃)', line=dict(color='#FFFF00'), legendgroup="4"), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], name='J值 (紫)', line=dict(color='#E066FF'), legendgroup="4"), 4, 1)

    fig.update_layout(template="plotly_dark", height=880, xaxis_rangeslider_visible=False, showlegend=True, margin=dict(r=180, t=50, b=50), legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, tracegroupgap=155, font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
        <div class='ai-advice-box'>
            <div class='confidence-tag'>{stock_accuracy}</div>
            <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
            <hr style='border:0.5px solid #444; margin:10px 0;'>
            <p><b>診斷：</b>{insight[1]} (乖離率: {bias:.2%})</p>
            <div style='background: #1C2128; padding: 12px; border-radius: 8px;'>
                <p style='color:#00F5FF; font-weight:bold;'>🔮 AI 統一展望 (基準日: {df.index[-1].strftime('%Y/%m/%d')} | 1,000次模擬)：</p>
                <p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估隔日收盤價：{insight[3]:.2f}</p>
                <p style='color:#8899A6;'>預估隔日浮動區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 6. 主程式 ---
# --- 6. 主程式 (完全對齊版) ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    if st.session_state.user and (time.time() - st.session_state.last_active > 3600): st.session_state.user = None
    st.session_state.last_active = time.time()
    
    # 建立一個簡單的快取，避免每秒都去刷 Google Sheets
    @st.cache_resource(ttl=30)
    def get_gsheets_connection():
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        return {
            "users": sh.worksheet("users"),
            "watchlist": sh.worksheet("watchlist"),
            "settings": sh.worksheet("settings"),
            "predictions": sh.worksheet("predictions")
        }

    try:
        sheets = get_gsheets_connection()
        ws_u, ws_w, ws_s, ws_p = sheets["users"], sheets["watchlist"], sheets["settings"], sheets["predictions"]
        
        # 讀取設定值
        s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
        cp = int(s_map.get('global_precision', 55))
        api_ttl = int(s_map.get('api_ttl_min', 1))
        tw_val = float(s_map.get('trend_weight', 1.0))
        v_comp = float(s_map.get('vol_comp', 1.5))
        
    except Exception as e:
        if "429" in str(e):
            st.error("🚨 Google API 請求過於頻繁，請等待 60 秒後手動重新整理頁面。")
        else:
            st.error(f"🚨 資料庫連線失敗: {e}")
        return

    # --- 登入邏輯 ---
    if st.session_state.user is None:
        st.title("🚀 StockAI 終端安全登入")
        tab_login, tab_reg = st.tabs(["🔑 帳號登入", "📝 申請權限"])
        
        with tab_login:
            u = st.text_input("管理帳號", key="login_u")
            p = st.text_input("存取密碼", type="password", key="login_p")
            if st.button("確認進入終端", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf.empty and not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("❌ 驗證失敗：帳號或密碼錯誤。")
        
        with tab_reg:
            st.markdown("#### 申請新的終端存取權限")
            new_u = st.text_input("欲設定的帳號", key="reg_u")
            new_p = st.text_input("欲設定的密碼", type="password", key="reg_p")
            confirm_p = st.text_input("再次確認密碼", type="password", key="reg_pc")
            if st.button("提交註冊申請"):
                if new_u and new_p == confirm_p:
                    udf = pd.DataFrame(ws_u.get_all_records())
                    if not udf.empty and new_u in udf['username'].astype(str).values:
                        st.error("⚠️ 此帳號已存在。")
                    else:
                        ws_u.append_row([str(new_u), str(new_p)])
                        st.success("✅ 註冊成功！")
                else:
                    st.warning("⚠️ 請檢查輸入資訊。")
    
    # --- 登入後的終端介面 ---
    else:
        with st.expander("⚙️ 終端設定面板", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("自選清單", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("➕ 快速新增 (代碼)")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("新增股票"):
                        if ns:
                            new_s = ns.upper().strip()
                            if new_s in u_stocks:
                                st.error(f"⚠️ {new_s} 已在清單中")
                            else:
                                ws_w.append_row([st.session_state.user, new_s])
                                st.success(f"✅ 已新增 {new_s}"); st.rerun()
                with c2:
                    if st.button("🗑️ 刪除目前選定"):
                        all_rows = ws_w.get_all_values()
                        for idx, row in reversed(list(enumerate(all_rows))):
                            if row[0] == st.session_state.user and row[1] == target:
                                ws_w.delete_rows(idx + 1)
                                break
                        st.success(f"✅ 已移除 {target}"); st.rerun()
            
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    temp_df, _ = fetch_comprehensive_data(target, api_ttl*60)
                    # 抓取 AI 優化數值
                    ai_res = auto_fine_tune_engine(temp_df, cp, tw_val, v_comp) if temp_df is not None else (cp, tw_val, v_comp, ("2330", "2382", "00878"), 0, 0)
                    ai_p, ai_tw, ai_v, ai_b = ai_res[0], ai_res[1], ai_res[2], ai_res[3]
                    
                    # 重新加入動態建議標籤
                    b1 = st.text_input(f"1. 權值標本 (AI 推薦: {ai_b[0]})", ai_b[0])
                    b2 = st.text_input(f"2. 成長標本 (AI 推薦: {ai_b[1]})", ai_b[1])
                    b3 = st.text_input(f"3. ETF 標本 (AI 推薦: {ai_b[2]})", ai_b[2])
                    new_p = st.slider(f"系統靈敏度 (AI 最優: {ai_p})", 0, 100, ai_p)
                    new_tw = st.number_input(f"AI 趨勢權重 (AI 最優: {ai_tw})", 0.5, 3.0, ai_tw)
                    new_ttl = st.number_input(f"API 快取控管 (建議 1-10 分鐘)", 1, 10, api_ttl)
                    new_v = st.slider(f"波動補償係數 (AI 最優: {ai_v})", 0.5, 3.0, ai_v)
                    
                    if st.button("💾 同步 AI 最優參數至雲端"):
                        ws_s.update_cell(2, 2, str(new_p)); ws_s.update_cell(3, 2, str(new_ttl))
                        ws_s.update_cell(4, 2, b1); ws_s.update_cell(5, 2, b2); ws_s.update_cell(6, 2, b3)
                        ws_s.update_cell(7, 2, str(new_tw)); ws_s.update_cell(8, 2, str(new_v))
                        st.success("✅ 參數同步成功！"); st.rerun()
                
                if st.button("🚪 登出系統"): 
                    st.session_state.user = None
                    st.rerun()
        
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

# 檔案最底部確保無縮排
if __name__ == "__main__": 
    main()










