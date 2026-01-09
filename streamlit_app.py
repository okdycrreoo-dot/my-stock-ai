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

# --- 1. 配置與 UI 視覺 (完整展開 CSS 以確保樣式精度) ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

st.markdown("""
    <style>
    .stApp { 
        background-color: #0E1117; 
        color: #FFFFFF !important; 
    }
    label, p, span, .stMarkdown, .stCaption { 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
    }
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

# --- 2. 數據引擎 (強化指標運算) ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_data(symbol, ttl_seconds):
    s_id = str(symbol).strip().upper()
    if not (s_id.endswith(".TW") or s_id.endswith(".TWO")): 
        s_id = f"{s_id}.TW"
    
    for attempt in range(3):
        try:
            # 抓取兩年數據以確保 60MA 與 MACD 穩定
            raw_df = yf.download(s_id, period="2y", interval="1d", auto_adjust=False, progress=False)
            if raw_df is not None and not raw_df.empty:
                if isinstance(raw_df.columns, pd.MultiIndex): 
                    raw_df.columns = raw_df.columns.get_level_values(0)
                
                # 均線系統
                raw_df['MA5'] = raw_df['Close'].rolling(5).mean()
                raw_df['MA20'] = raw_df['Close'].rolling(20).mean()
                raw_df['MA60'] = raw_df['Close'].rolling(60).mean()
                
                # MACD 能量指標
                exp12 = raw_df['Close'].ewm(span=12, adjust=False).mean()
                exp26 = raw_df['Close'].ewm(span=26, adjust=False).mean()
                raw_df['MACD'] = exp12 - exp26
                raw_df['Signal'] = raw_df['MACD'].ewm(span=9, adjust=False).mean()
                raw_df['Hist'] = raw_df['MACD'] - raw_df['Signal']
                
                # KDJ 隨機指標
                low_9 = raw_df['Low'].rolling(9).min()
                high_9 = raw_df['High'].rolling(9).max()
                rsv = (raw_df['Close'] - low_9) / (high_9 - low_9 + 0.001) * 100
                raw_df['K'] = rsv.ewm(com=2).mean()
                raw_df['D'] = raw_df['K'].ewm(com=2).mean()
                raw_df['J'] = 3 * raw_df['K'] - 2 * raw_df['D']
                
                return raw_df.dropna(), s_id
            time.sleep(1.5)
        except Exception as e:
            time.sleep(1.5)
            continue
    return None, s_id
# --- 3. 背景自動對帳與命中率反饋系統 (對齊 okdycrreoo 管理邏輯) ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        # 1. 讀取 Google Sheets 所有紀錄並轉為處理對象
        raw_records = ws_p.get_all_records()
        df_history = pd.DataFrame(raw_records)
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 2. 自動回填邏輯：檢查過去是否有未結算實際價格的預測
        for idx, row in df_history.iterrows():
            is_empty = str(row['actual_close']) == ""
            is_not_today = row['date'] != today_date
            
            if is_empty and is_not_today:
                # 抓取該預測日期的實體收盤價 (擴大檢索範圍至 3 日以避免週末)
                check_start = row['date']
                check_end = (pd.to_datetime(row['date']) + timedelta(days=3)).strftime("%Y-%m-%d")
                history_df = yf.download(row['symbol'], start=check_start, end=check_end, progress=False)
                
                if not history_df.empty:
                    real_val = float(history_df['Close'].iloc[0])
                    # 更新 Google Sheets：第 6 欄填入實際價，第 7 欄填入誤差百分比
                    ws_p.update_cell(idx + 2, 6, round(real_val, 2))
                    error_rate = (real_val - float(row['pred_close'])) / float(row['pred_close'])
                    ws_p.update_cell(idx + 2, 7, f"{error_rate:.2%}")

        # 3. 每日存檔機制：若今日尚未記錄，則將當前 AI 預測結果上傳雲端
        already_logged = any((r['date'] == today_date and r['symbol'] == f_id) for r in raw_records)
        if not already_logged:
            # 格式：日期, 代碼, 預估價, 區間低, 區間高, 實際價(待填), 誤差(待填)
            archive_data = [today_date, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""]
            ws_p.append_row(archive_data)
        
        # 4. 命中率統計：分析該股票最近 10 次預測的區間精準度
        df_target = df_history[(df_history['symbol'] == f_id) & (df_history['actual_close'] != "")].tail(10)
        if not df_target.empty:
            within_range = (df_target['actual_close'] >= df_target['range_low']) & (df_target['actual_close'] <= df_target['range_high'])
            final_rate = (sum(within_range) / len(df_target)) * 100
            return f"🎯 此股實戰命中率: {final_rate:.1f}%"
        return "🎯 數據累積中"
    except Exception as e:
        return f"🎯 雲端同步中"

# --- 4. AI 核心：三大腦進化微調引擎 (均值回歸/量價加權/波動融合) ---
def auto_fine_tune_engine(df, base_p, base_tw, v_comp):
    # A. 數據基礎準備
    returns = df['Close'].pct_change().dropna()
    price_now = float(df['Close'].iloc[-1])
    
    # B. 進化一：波動融合 (Fused Volatility) - 綜合短中長期的市場震盪幅度
    vol_5 = returns.tail(5).std()
    vol_20 = returns.tail(20).std()
    vol_60 = returns.tail(60).std()
    f_vol = (vol_5 * 0.5) + (vol_20 * 0.3) + (vol_long * 0.2)
    
    # C. 進化二：量價加權 (Volume Weighting) - 爆量時自動放大趨勢權重
    v_curr = df['Volume'].iloc[-1]
    v_avg5 = df['Volume'].tail(5).mean()
    vol_spike = v_curr / (v_avg5 + 0.1)
    f_tw = max(0.5, min(2.5, 1.0 + (returns.tail(5).mean() * 15 * min(1.5, vol_spike))))
    
    # D. 進化三：均值回歸 (Mean Reversion) - 計算股價與 20 日均線的乖離程度
    ma20_val = df['Close'].rolling(20).mean().iloc[-1]
    bias_val = (price_now - ma20_val) / (ma20_val + 0.1)
    
    # E. 根據 AI 學習結果決定靈敏度與補償
    f_p = 45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60
    atr_approx = (df['High'] - df['Low']).tail(5).mean() / price_now
    f_vc = 1.3 if atr_approx > 0.035 else 2.1 if atr_approx < 0.015 else 1.7
    
    # F. AI 推薦標本自動選取
    recs = ("2330", "2382", "00878") if f_vol > 0.015 else ("2317", "2454", "0050")
    
    return int(f_p), round(f_tw, 2), f_vc, recs, bias_val, f_vol

# --- 5. 蒙地卡羅模擬運算引擎 (Monte Carlo Simulation) ---
def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol):
    last_close = float(df['Close'].iloc[-1])
    last_open = float(df['Open'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    last_vol = int(df['Volume'].iloc[-1])
    
    # 設置隨機種子確保可重複性
    np.random.seed(42)
    all_paths = []
    
    # 基於靈敏度的核心趨勢率
    core_drift = ((int(precision) - 55) / 1000) * float(trend_weight)
    
    for _ in range(1000):
        # 產生未來 p_days 天的隨機擾動
        shocks = np.random.normal(0, f_vol * v_comp, p_days)
        path = [last_close]
        for i in range(p_days):
            # 關鍵：將均值回歸拉力 (Bias * 0.08) 注入漂移項
            reversion_pull = bias * 0.08
            daily_drift = core_drift - reversion_pull
            next_price = path[-1] * (1 + daily_drift + shocks[i])
            path.append(next_price)
        all_paths.append(path[1:])
    
    # 計算預測平均值與信賴區間
    pred_seq = np.mean(all_paths, axis=0)
    # 此處邏輯延伸至第三段渲染
# 接續蒙地卡羅運算：計算首日預測分布與信賴區間
    first_day_dist = [p[0] for p in all_paths]
    std_dev = np.std(first_day_dist)
    next_close = pred_seq[0]
    
    # 均線回歸與支撐壓力位計算 (基於 AI 靈敏度)
    sensitivity = (int(precision) / 55)
    periods = {"5日短期": (df['MA5'].iloc[-1], 0.8), 
               "20日中期": (df['MA20'].iloc[-1], 1.5), 
               "60日長期": (df['MA60'].iloc[-1], 2.2)}
    
    advice = {}
    for label, (ma_val, factor) in periods.items():
        # 波動融合應用於買賣建議區間
        buy_p = ma_val * (1 - f_vol * v_comp * factor * sensitivity)
        sell_p = ma_val * (1 + f_vol * v_comp * factor * sensitivity)
        advice[label] = {"buy": buy_p, "sell": sell_p}
    
    # AI 形態診斷分數
    score = 0
    reasons = []
    if last_close > df['MA20'].iloc[-1]: 
        score += 1; reasons.append("股價站上月線")
    else: 
        score -= 1; reasons.append("股價跌破月線")
        
    if df['Hist'].iloc[-1] > 0: 
        score += 1; reasons.append("MACD 多頭放量")
    if df['K'].iloc[-1] < 25: 
        score += 1; reasons.append("KDJ 低檔超賣")
    
    status_map = {2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 
                  0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41")}
    res_text, res_color = status_map.get(score if score in status_map else -1, ("📉 偏空警戒", "#00FF41"))
    
    # 輸出封裝：包含漲跌幅計算與完整診斷
    chg_pct = ((last_close - prev_close) / prev_close) * 100
    insight_package = (res_text, " | ".join(reasons), res_color, next_close, next_close + (std_dev * 1.5), next_close - (std_dev * 1.5))
    
    return pred_seq, advice, last_close, last_open, prev_close, last_vol, chg_pct, insight_package

# --- 6. 終端渲染 UI ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    # 調用數據與進化引擎
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗，請檢查代碼或網路。")
        return

    # 執行 AI 進化邏輯獲取最優參數
    final_p, final_tw, ai_v, _, bias, f_vol = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, chg_pct, insight = perform_ai_engine(df, p_days, final_p, final_tw, ai_v, bias, f_vol)
    
    # 雲端對帳同步
    accuracy_label = auto_sync_feedback(ws_p, f_id, insight)

    # 頂部儀表板渲染
    st.title(f"📊 {f_id} 實戰全能終端")
    st.caption(f"✨ AI 大腦已接管：靈敏度 {final_p} | 趨勢權重 {final_tw} | 波動融合 {f_vol:.4f} | 乖離率 {bias:.2%}")

    price_color = "#FF3131" if chg_pct >= 0 else "#00FF41"
    m_cols = st.columns(5)
    metrics = [
        ("當前價格", f"{curr_p:.2f}", price_color),
        ("今日漲跌", f"{chg_pct:+.2f}%", price_color),
        ("今日開盤", f"{open_p:.2f}", "#FFFFFF"),
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"),
        ("今日成交 (張)", f"{int(curr_v/1000):,}", "#FFFF00")
    ]
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: 
            st.markdown(f"<div class='info-box'><span class='label-text'>{lab}</span><span class='realtime-val' style='color:{col}'>{val}</span></div>", unsafe_allow_html=True)

    # 支撐壓力建議區
    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: 
            st.markdown(f"<div class='diag-box'><center><b>{label}</b></center><hr style='border:0.5px solid #444'>買入建議: <span class='price-buy'>{p['buy']:.2f}</span><br>賣出建議: <span class='price-sell'>{p['sell']:.2f}</span></div>", unsafe_allow_html=True)

    # Plotly 四層圖表建構
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04,
                        subplot_titles=("價格與 AI 預測路徑", "成交量分析 (張)", "MACD 能量柱", "KDJ 擺動指標"))
    
    p_df = df.tail(90)
    # 1. K線與預測線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='實體 K 線'), 1, 1)
    f_dates = [p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)]
    fig.add_trace(go.Scatter(x=f_dates, y=pred_line, name='AI 模擬路徑', line=dict(color='#FF3131', width=3, dash='dash')), 1, 1)
    
    # 2. 成交量
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume']/1000, name='成交量', marker_color=v_colors), 2, 1)
    
    # 3. MACD
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], name='MACD 力道', marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']]), 3, 1)
    
    # 4. KDJ
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], name='K 值', line=dict(color='#00F5FF')), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], name='D 值', line=dict(color='#FFFF00')), 4, 1)
    
    fig.update_layout(template="plotly_dark", height=880, xaxis_rangeslider_visible=False, showlegend=True, margin=dict(r=10, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    
    # AI 診斷結論盒
    st.markdown(f"""
        <div class='ai-advice-box'>
            <div class='confidence-tag'>{accuracy_label}</div>
            <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
            <hr style='border:0.5px solid #444; margin:10px 0;'>
            <p><b>技術面診斷：</b>{insight[1]}</p>
            <div style='background: #1C2128; padding: 12px; border-radius: 8px;'>
                <p style='color:#00F5FF; font-weight:bold;'>🔮 AI 隔日展望：</p>
                <p style='font-size:1.3rem; color:#FFAC33; font-weight:900;'>預估收盤價：{insight[3]:.2f}</p>
                <p style='color:#8899A6;'>波動預期區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 7. 主程式邏輯與 okdycrreoo 管理面控制 ---
def main():
    if 'user' not in st.session_state: 
        st.session_state.user, st.session_state.last_active = None, time.time()
    
    # 自動登出邏輯 (10分鐘)
    if st.session_state.user and (time.time() - st.session_state.last_active > 600):
        st.session_state.user = None
    st.session_state.last_active = time.time()
    
    try:
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        sh = gspread.authorize(creds).open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        ws_u, ws_w, ws_s, ws_p = sh.worksheet("users"), sh.worksheet("watchlist"), sh.worksheet("settings"), sh.worksheet("predictions")
    except Exception:
        st.error("🚨 資料庫連線失敗，請檢查 Secret 配置。"); return

    # 獲取雲端參數 (含 API 頻率控制 1~10 分鐘)
    s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
    cp = int(s_map.get('global_precision', 55))
    api_ttl = int(s_map.get('api_ttl_min', 1))
    tw_val = float(s_map.get('trend_weight', 1.0))
    v_comp = float(s_map.get('vol_comp', 1.5))

    if st.session_state.user is None:
        st.title("🚀 StockAI 登入系統")
        u = st.text_input("管理帳號", key="login_u")
        p = st.text_input("密碼", type="password", key="login_p")
        if st.button("確認進入終端", use_container_width=True):
            udf = pd.DataFrame(ws_u.get_all_records())
            if not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                st.session_state.user = u; st.rerun()
            else: st.error("驗證失敗：帳號或密碼錯誤")
    else:
        with st.expander("⚙️ 終端管理與 AI 設定面板", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("監測股票選單", u_stocks if u_stocks else ["2330"])
                ns = st.text_input("➕ 新增監測代碼 (例如: 2454)")
                if st.button("執行新增"): 
                    if ns: ws_w.append_row([st.session_state.user, ns.upper()]); st.rerun()
                if st.button("🗑️ 移除目前選定"):
                    all_rows = ws_w.get_all_values()
                    for idx, row in enumerate(all_rows):
                        if row[0] == st.session_state.user and row[1] == target:
                            ws_w.delete_rows(idx + 1); st.rerun()
            with col_b:
                p_days = st.number_input("AI 預測跨度 (天)", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("### 🛠️ 管理員戰情室")
                    # 計算 AI 建議值以供參考
                    t_df, _ = fetch_comprehensive_data(target, api_ttl*60)
                    ai_p, ai_tw, ai_v, ai_b, _, _ = auto_fine_tune_engine(t_df, cp, tw_val, v_comp) if t_df is not None else (cp, tw_val, v_comp, ("2330", "2317", "0050"), 0, 0)
                    
                    b1 = st.text_input(f"1. 權值基準 (AI 推薦: {ai_b[0]})", ai_b[0])
                    b2 = st.text_input(f"2. 成長基準 (AI 推薦: {ai_b[1]})", ai_b[1])
                    b3 = st.text_input(f"3. 指標 ETF (AI 推薦: {ai_b[2]})", ai_b[2])
                    
                    if st.button("💾 同步 AI 優化參數至雲端"):
                        ws_s.update_cell(2, 2, str(ai_p))    # 靈敏度
                        ws_s.update_cell(7, 2, str(ai_tw))   # 趨勢權重
                        ws_s.update_cell(8, 2, str(ai_v))    # 波動補償
                        ws_s.update_cell(4, 2, b1)
                        ws_s.update_cell(5, 2, b2)
                        ws_s.update_cell(6, 2, b3)
                        st.success("✅ AI 最佳化參數已同步至 Google Sheets！"); st.rerun()
                if st.button("🚪 安全登出"): 
                    st.session_state.user = None; st.rerun()
        
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

if __name__ == "__main__": 
    main()
