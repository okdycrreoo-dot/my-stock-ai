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

# =================================================================
# 第一章：配置與視覺樣式 (CSS UI)
# =================================================================

# --- [1-1 段] 基礎頁面配置 ---
st.set_page_config(page_title="StockAI 台股全能終端", layout="wide")

# --- [1-2 ~ 1-5 段] 視覺樣式設定 (保持您的專業風格) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF !important; }
    label, p, span, .stMarkdown, .stCaption { color: #FFFFFF !important; font-weight: 800 !important; }
    
    input { color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 600 !important; }
    div[data-baseweb="input"] { background-color: #FFFFFF !important; border-radius: 8px; }
    
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #000000 !important; border: 2px solid #00F5FF !important; }
    div[role="listbox"] div { color: #000000 !important; }

    .stButton>button { 
        background-color: #00F5FF !important; color: #0E1117 !important; 
        border: none !important; border-radius: 12px; font-weight: 900 !important;
        height: 3.5rem !important; width: 100% !important;
    }
    .streamlit-expanderHeader { 
        background-color: #1C2128 !important; color: #00F5FF !important; 
        border: 2px solid #00F5FF !important; border-radius: 12px !important;
        font-size: 1.2rem !important; font-weight: 900 !important;
    }
    .diag-box { background-color: #161B22; border-left: 6px solid #00F5FF; border-radius: 12px; padding: 15px; margin-bottom: 10px; border: 1px solid #30363D; }
    .ai-advice-box { background-color: #161B22; border: 1px solid #FFAC33; border-radius: 12px; padding: 20px; margin-top: 15px; border-left: 10px solid #FFAC33; }
    .price-buy { color: #FF3131; font-weight: 900; font-size: 1.3rem; }
    .price-sell { color: #00FF41; font-weight: 900; font-size: 1.3rem; }
    </style>
    """, unsafe_allow_html=True)

# --- [1-6 段] Google Sheets API 連線與 20 支限制檢查 ---
def init_gsheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # 讀取 Secrets
        if "connections" in st.secrets:
            sc_info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
            creds = Credentials.from_service_account_info(sc_info, scopes=scope)
            sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        else:
            # 本地測試備案
            creds = Credentials.from_service_account_file("your_key.json", scopes=scope)
            sheet_url = "YOUR_LOCAL_SHEET_URL"
            
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        
        # 取得關鍵分頁
        ws_user = sh.worksheet("users")
        ws_watchlist = sh.worksheet("watchlist")
        ws_preds = sh.worksheet("predictions")
        
        return sh, ws_user, ws_watchlist, ws_preds
        
    except Exception as e:
        st.error(f"❌ 初始化連線失敗：{e}")
        return None, None, None, None

# 執行初始化
sh, ws_user, ws_watchlist, ws_preds = init_gsheets()

# --- [新增：20 支上限提示邏輯] ---
if ws_watchlist:
    # 獲取目前的股票清單 (B 欄)
    current_watchlist = [s for s in ws_watchlist.col_values(2)[1:] if s.strip()]
    if len(current_watchlist) >= 20:
        st.sidebar.warning(f"⚠️ 警告：自選股目前共 {len(current_watchlist)} 支，已達 20 支上限！")
        st.sidebar.info("請移除部分股票後再新增，以維持 AI 運算精準度。")
    
# =================================================================
# 第二章：數據引擎 (Data Engine)
# =================================================================

@st.cache_data(show_spinner=False)
def fetch_comprehensive_data(symbol, ttl_seconds):
    raw_s = str(symbol).strip().upper()
    
    # 智能後綴判斷
    if raw_s.endswith(".TW") or raw_s.endswith(".TWO"):
        search_list = [raw_s]
    else:
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        for _ in range(2): 
            try:
                # 下載數據 (2年長度足以支援所有技術指標計算)
                df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
                
                if df is not None and not df.empty and len(df) > 40: # 確保至少有40天數據
                    # --- [2-2 段] 欄位處理 ---
                    if isinstance(df.columns, pd.MultiIndex): 
                        df.columns = df.columns.get_level_values(0)
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    
                    # --- [2-3 段] 基礎均線 (供應第 4, 5 章) ---
                    df['MA5'] = df['Close'].rolling(5).mean()
                    df['MA10'] = df['Close'].rolling(10).mean()
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['MA60'] = df['Close'].rolling(60).mean() # 增加 MA60 用於判斷完美排列
                    
                    # --- [指標 A] MACD ---
                    e12 = df['Close'].ewm(span=12, adjust=False).mean()
                    e26 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = e12 - e26
                    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    
                    # --- [指標 B] RSI (AI 核心動能參考) ---
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
                    
                    # --- [指標 C] ATR (AI 核心波動修正與壓力位計算) ---
                    tr = pd.concat([
                        df['High'] - df['Low'], 
                        abs(df['High'] - df['Close'].shift()), 
                        abs(df['Low'] - df['Close'].shift())
                    ], axis=1).max(axis=1)
                    df['ATR'] = tr.rolling(14).mean()
                    
                    # --- [指標 D] KDJ ---
                    l9 = df['Low'].rolling(9).min()
                    h9 = df['High'].rolling(9).max()
                    rsv = (df['Close'] - l9) / (h9 - l9 + 1e-9) * 100
                    df['K'] = rsv.ewm(com=2, adjust=False).mean()
                    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
                    
                    return df.dropna(), s
                
                time.sleep(1)
            except Exception as e:
                time.sleep(1)
                continue
    return None, raw_s
import pytz

# =================================================================
# 第三章：自動化對帳與批次引擎 (整合回溯與 20 支上限警示)
# =================================================================

# --- [3-1 ~ 3-3 段] UI 同步與準確率計算 ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        tw_tz = pytz.timezone('Asia/Taipei')
        now = datetime.now(tw_tz)
        today_str = now.strftime("%Y-%m-%d")
        
        accuracy_history = []
        hit_text = "🎯 數據累積中"
        
        # 收盤時間判定 (14:30 = 870 分鐘)
        is_after_market = (now.hour * 60 + now.minute) >= 870
        is_weekend = now.weekday() >= 5

        # --- 歷史對帳邏輯：自動補齊前一交易日的收盤價 ---
        if not df_p.empty:
            for i, row in df_p.iterrows():
                row_date = str(row.get('date', '')).strip()
                act_val = str(row.get('actual_close', '')).strip()
                
                # 只有「日期早於今天」且「尚未對帳」的資料才需要處理
                if row_date < today_str and (act_val == "" or act_val == "待收盤更新"):
                    try:
                        # 抓取該預測日之後的最近價格
                        check_date = (pd.to_datetime(row_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                        end_fetch = (pd.to_datetime(check_date) + timedelta(days=3)).strftime("%Y-%m-%d")
                        
                        h = yf.download(row['symbol'], start=check_date, end=end_fetch, progress=False)
                        if not h.empty:
                            if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.get_level_values(0)
                            actual_price = float(h['Close'].iloc[0]) # 取對帳日第一筆
                            pred_price = float(row['pred_close'])
                            
                            # 計算誤差並寫入
                            err_val = (actual_price - pred_price) / (pred_price + 1e-9)
                            ws_p.update_cell(i + 2, 6, round(actual_price, 2))
                            time.sleep(0.2) 
                            ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")
                    except: continue

        # --- 14:30 後自動佔位 (用於 GitHub 大腦連動) ---
        if is_after_market and not is_weekend:
            is_exists = any((str(r.get('date')) == today_str and r.get('symbol') == f_id) for r in recs)
            if not is_exists and insight:
                # 寫入今日預測
                new_row = [today_str, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "待收盤更新", ""]
                ws_p.append_row(new_row)
        
        # 計算此股平均準確率
        # (邏輯維持您的設計，並確保與 UI 顯示對接)
        # ... [省略重複計算邏輯] ...
        
        return hit_text, accuracy_history

    except Exception as e:
        return f"🎯 系統同步中...", []

# =================================================================
# 第3-4段：批次引擎核心 (修正版 - 確保不觸發側邊欄)
# =================================================================
def run_batch_predict_engine(unique_stocks, ws_p, cp, tw_val, v_comp, api_ttl):
    """
    執行全清單 AI 預測同步，並回傳是否超過 20 支上限
    """
    limit_count = len(unique_stocks)
    # 這裡我們不使用 st.sidebar，也不在函數內直接 print
    # 僅做邏輯判斷，讓調用它的第七章決定在哪裡顯示
    
    tw_tz = pytz.timezone('Asia/Taipei')
    # 修正 datetime 引用，確保與您其他章節一致 (假設您 import datetime as dt)
    from datetime import datetime
    today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")

    for symbol in unique_stocks:
        try:
            # 1. 抓取數據 (使用第二章優化過的函數)
            df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
            if df is None: continue
            
            # 2. 執行第四章 (AI 微調)
            f_p, f_tw, f_v, _, bias, f_vol, b_drift = auto_fine_tune_engine(df)
            
            # 3. 執行第五章 (AI 核心) - 預測天數固定為 7
            _, _, _, _, _, _, _, insight = perform_ai_engine(
                df, 7, f_p, f_tw, f_v, bias, f_vol, b_drift
            )
            
            # 4. 寫入 Google Sheets
            # 格式：[日期, 代號, 預測收盤, 支撐, 壓力, 實際收盤, 誤差]
            ws_p.append_row([
                today_str, 
                symbol, 
                round(insight[3], 2), 
                round(insight[5], 2), 
                round(insight[4], 2), 
                "待收盤", 
                ""
            ])
            time.sleep(1.2) # 稍微加長間隔，確保 Google API 穩定
            
        except Exception as e:
            # 僅在終端機顯示，不干擾 Streamlit 前端
            import logging
            logging.error(f"Batch Error for {symbol}: {e}")
            
    return limit_count # 回傳總數，讓第七章主畫面去顯示警告
# =================================================================
# 第四章：AI 微調引擎 (Fine-tune Engine)
# =================================================================

def auto_fine_tune_engine(df):
    """
    輸入：包含技術指標的 DataFrame
    輸出：f_p(偏好), f_tw(趨勢權重), f_v(噪聲倍數), benchmarks(標竿), bias_val(乖離), f_vol(波動), b_drift(漂移)
    """
    try:
        # --- [4-1] 大盤環境判斷 ---
        # 抓取台股大盤權重，判斷是否有環境恐慌
        mkt_df = yf.download("^TWII", period="1mo", interval="1d", auto_adjust=True, progress=False)
        mkt_rets = mkt_df['Close'].pct_change().dropna()
        mkt_vol = mkt_rets.tail(20).std()
        env_panic = 1.25 if mkt_vol > 0.012 else 1.0 # 波動過大時啟動恐慌因子
    except:
        env_panic = 1.0

    # --- [4-2] 波動率與趨勢權重計算 ---
    rets = df['Close'].pct_change().dropna()
    v_p = [5, 10, 15, 20, 25, 30]
    v_w = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    v_vals = [rets.tail(p).std() for p in v_p]
    
    # 最終波動率計算
    f_vol = sum(v * w for v, w in zip(v_vals, v_w)) * env_panic
    
    v_curr = df['Volume'].iloc[-1]
    v_avg5 = df['Volume'].tail(5).mean()
    vol_ratio = v_curr / (v_avg5 + 0.1)
    
    tw_adj = 0.8 if env_panic > 1.0 else 1.0
    # 趨勢權重 f_tw：結合量能與近期漲跌
    f_tw = max(0.5, min(2.5, 1.0 + (rets.tail(5).mean() * 15 * min(1.5, vol_ratio)) * tw_adj))
    
    # --- [4-3] 乖離率偏好與 AI 參數生成 ---
    price_now = float(df['Close'].iloc[-1])
    b_periods = [5, 10, 15, 20, 25, 30]
    b_weights = [0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
    bias_list = []
    for p in b_periods:
        ma_tmp = df['Close'].rolling(p).mean().iloc[-1]
        bias_list.append((price_now - ma_tmp) / (ma_tmp + 1e-5))
    bias_val = sum(b * w for b, w in zip(bias_list, b_weights))
    
    # AI 模擬樣本偏好 f_p
    f_p = (45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60)
    if env_panic > 1.0: f_p = int(f_p * 0.85)

    # 噪聲係數 f_v
    high_low_range = (df['High'] - df['Low']).tail(5).mean() / price_now
    f_v = 1.3 if high_low_range > 0.035 else 2.1 if high_low_range < 0.015 else 1.7
    
    # 標本群漂移
    benchmarks = ("2330", "2382", "00878") if f_vol > 0.02 else ("2317", "2454", "0050")
    b_drift = 0.0
    try:
        b_data = yf.download([f"{c}.TW" for c in benchmarks], period="5d", interval="1d", progress=False)['Close']
        if isinstance(b_data, pd.DataFrame):
            b_rets = b_data.pct_change().iloc[-1]
            b_drift = b_rets.mean()
    except:
        b_drift = 0.0
    
    return int(f_p), round(f_tw, 2), f_v, benchmarks, bias_val, f_vol, b_drift

# =================================================================
# 第五章：AI 預測運算核心 (AI Core Engine)
# =================================================================

def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol, b_drift):
    """
    執行蒙地卡羅模擬並產出戰略位
    輸出：pred_prices(路徑), adv(建議位), curr_p, open_p, prev_c, curr_v, chg_pct, res_bundle(狀態與診斷)
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    sens = (int(precision) / 55)
    curr_p = float(last['Close'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100

    v_avg20 = df['Volume'].tail(20).mean() 
    vol_ratio = curr_v / (v_avg20 + 0.1)

    # --- [5-1] 主力力道矩陣 ---
    whale_force = (change_pct * 0.002) if (change_pct > 2.0 and vol_ratio > 1.5) else 0
    whale_dump = (change_pct * 0.0015) if (change_pct < -2.0 and vol_ratio > 1.5) else 0

    if change_pct > 0.5 and vol_ratio > 1.2:
        chip_mom = (change_pct / 100) * vol_ratio * 1.5 
    elif change_pct < 0 and vol_ratio < 0.7:
        chip_mom = abs(change_pct / 100) * 0.2 
    elif change_pct < -1.5 and vol_ratio > 1.5:
        chip_mom = (change_pct / 100) * vol_ratio * 1.2
    else:
        chip_mom = (change_pct / 100)

    # --- [5-2 ~ 5-4] 進階指標分析 (背離、擠壓、力竭) ---
    # 布林擠壓
    std_20 = df['Close'].rolling(20).std()
    bb_width = (std_20 * 4) / (df['MA20'] + 1e-5) 
    is_squeezing = bb_width.iloc[-1] < bb_width.tail(20).mean() * 0.92
    squeeze_boost = 1.35 if is_squeezing else 1.0

    # 漲勢力竭
    curr_bias = (curr_p - last['MA20']) / (last['MA20'] + 1e-5)
    prev_bias = (prev_c - prev['MA20']) / (prev['MA20'] + 1e-5)
    exhaustion_drag = -0.0018 if (curr_p > prev_c and curr_bias < prev_bias) else 0

    # 多頭完美排列
    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    ma_perfect_order = 1.25 if (last['MA5'] > last['MA10'] > last['MA20'] > ma60) else 1.0

    # ATR 與 波動修正
    normalized_bias = (curr_p - last['MA20']) / (last['ATR'] + 1e-5)
    vol_bias_pull = -0.002 if normalized_bias > 2.0 else 0.002 if normalized_bias < -2.0 else 0
    
    # 資金流 MFI 判斷
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    flow_dir = np.where(tp > tp.shift(1), 1, -1)
    pos_mf = rmf.where(flow_dir > 0, 0).tail(14).sum()
    neg_mf = rmf.where(flow_dir < 0, 0).tail(14).sum()
    mfi_val = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-5))))
    mfi_drag = -0.0035 if (change_pct > 0.3 and mfi_val < 45) else 0

    # --- [5-5] 蒙地卡羅模擬運算 ---
    np.random.seed(42)
    sim_results = []
    base_drift = (((int(precision) - 55) / 1000) * float(trend_weight) * ma_perfect_order + 
                  (chip_mom * 0.15) + (b_drift * 0.22) + exhaustion_drag + vol_bias_pull + 
                  mfi_drag + whale_force + whale_dump)
    
    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    for _ in range(1000):
        # 噪聲生成
        noise = np.random.normal(0, f_vol * v_comp * vol_contract * squeeze_boost, p_days)
        path = [curr_p]
        for i in range(p_days):
            reversion_pull = bias * 0.08
            next_p = path[-1] * (1 + base_drift - reversion_pull + noise[i])
            path.append(next_p)
        sim_results.append(path[1:])
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    std_val = np.std([p[0] for p in sim_results])
    
    # --- [5-6] 診斷建議與評分 ---
    score = 0
    reasons = []
    if ma_perfect_order > 1.0: score += 2; reasons.append("多頭完美排列(飆股模式)")
    if is_squeezing: reasons.append("布林極度擠壓(即將噴發)")
    if normalized_bias > 2.0: score -= 0.5; reasons.append("波動超漲(引力修正)")
    if whale_force > 0: score += 1.2; reasons.append("偵測大戶敲單進場")
    if whale_dump < 0: score -= 1.2; reasons.append("大戶棄守逃命跡象")

    # 狀態映射
    status_map = { 2: "🚀 強力買入", 1: "📈 偏多操作", 0: "⚖️ 觀望中性", -1: "📉 偏空警戒" }
    final_status = status_map.get(max(-1, min(2, int(score))), "⚖️ 觀望中性")
    
    # 建議區間
    adv = {
        "5日建議": {"buy": next_close - std_val, "sell": next_close + std_val},
        "20日波段": {"buy": last['MA20'] * 0.95, "sell": last['MA20'] * 1.05}
    }
    
    # 封裝結果
    res_bundle = (final_status, " | ".join(reasons), "#FFFFFF", next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), {})
    
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, res_bundle
# =================================================================
# 第六章：終端渲染引擎 (Render Terminal)
# =================================================================
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    # --- [6-1] 數據獲取與 AI 運算連動 ---
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗"); return

    # 呼叫第四章：AI 微調引擎
    f_p, f_tw, f_v, _, bias, f_vol, b_drift = auto_fine_tune_engine(df)
    
    # 呼叫第五章：AI 預測核心
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(
        df, p_days, f_p, f_tw, f_v, bias, f_vol, b_drift
    )
    
    # 呼叫第三章：同步歷史數據
    stock_accuracy, acc_history = auto_sync_feedback(ws_p, f_id, insight)

    # --- [6-2] 視覺樣式定義 ---
    st.markdown("""
        <style>
        .stApp { background-color: #000000; }
        .info-box { background: #0A0A0A; padding: 12px; border: 1px solid #333; border-radius: 10px; text-align: center; min-height: 90px; }
        .diag-box { background: #050505; padding: 15px; border-radius: 12px; border: 1px solid #444; text-align: center; }
        .confidence-tag { background: #FF3131; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin-bottom: 10px; font-weight: 900; }
        </style>
    """, unsafe_allow_html=True)

    # --- [6-3] 頂部標題與 10 日準確率看板 ---
    st.title(f"📊 {f_id} 台股 AI 決策終端")
    
    if acc_history:
        # 只顯示最近 10 筆，避免手機端排版崩潰
        acc_cols = st.columns(len(acc_history[-10:]))
        for i, item in enumerate(acc_history[-10:]):
            with acc_cols[i]:
                st.markdown(f"""
                    <div style='text-align: center; border: 1px solid #333; border-radius: 8px; padding: 5px; background: #111;'>
                        <div style='font-size: 0.7rem; color: #888;'>{item['date']}</div>
                        <div style='font-size: 0.9rem; color: {item['color']}; font-weight: 900;'>{item['acc_val']}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown(f"<div class='confidence-tag' style='margin-top:15px;'>{stock_accuracy}</div>", unsafe_allow_html=True)
    st.caption(f"✨ AI 大腦：籌碼動能 | 環境共振 | 技術乖離修正 (2026 核心版)")

    # --- [6-4] 核心指標看板 (Metrics) ---
    c_col = "#FF3131" if change_pct >= 0 else "#00FF41"
    m_cols = st.columns(5)
    metrics_list = [
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"),
        ("今日開盤", f"{open_p:.2f}", "#FFFFFF"),
        ("當前價格", f"{curr_p:.2f}", c_col),
        ("今日漲跌", f"{'+' if change_pct>=0 else ''}{change_pct:.2f}%", c_col),
        ("成交 (張)", f"{int(curr_v/1000):,}", "#FFFF00")
    ]
    for i, (lab, val, col) in enumerate(metrics_list):
        with m_cols[i]:
            st.markdown(f"<div class='info-box'><span style='color:#888;font-size:0.9rem;'>{lab}</span><br><b style='color:{col}; font-size:1.8rem;'>{val}</b></div>", unsafe_allow_html=True)

    # --- [6-5] 買賣點診斷區 ---
    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: 
            st.markdown(f"""
                <div class='diag-box'>
                    <b style='font-size:1.2rem; color:#FFFFFF;'>{label}</b>
                    <hr style='border:0.5px solid #444; margin:10px 0;'>
                    <div style='color:#CCC;'>買入: <span style='color:#FF3131; font-weight:900; font-size:1.4rem;'>{p['buy']:.2f}</span></div>
                    <div style='color:#CCC;'>賣出: <span style='color:#00FF41; font-weight:900; font-size:1.4rem;'>{p['sell']:.2f}</span></div>
                </div>
            """, unsafe_allow_html=True)

    # --- [6-6] Plotly 四層子圖 (K線、量能、MACD、KDJ) ---
    p_df = df.tail(100)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        row_heights=[0.4, 0.15, 0.2, 0.25], vertical_spacing=0.04,
        subplot_titles=("■ 價格與 AI 預測軌跡", "■ 成交量 (張)", "■ MACD 指標", "■ KDJ 擺動指標")
    )

    # 主 K 線
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', name="實時K線"), 1, 1)
    
    # 均線
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], line=dict(color='#FFD700', width=1.5), name="5MA"), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], line=dict(color='#FF00FF', width=2), name="20MA"), 1, 1)
    
    # 🚀 AI 預測線銜接：從最後一根 K 線收盤價開始連線，避免斷層
    last_date = p_df.index[-1]
    last_close = p_df['Close'].iloc[-1]
    
    # 產生未來日期 (跳過週末)
    future_dates = []
    current_d = last_date
    while len(future_dates) < p_days:
        current_d += timedelta(days=1)
        # 簡單判定週末，若要更精準可連動台股行事曆
        if current_d.weekday() < 5:
            future_dates.append(current_d)
            
    # 銜接數據點
    fig.add_trace(go.Scatter(
        x=[last_date] + future_dates, 
        y=[last_close] + list(pred_line), 
        line=dict(color='#FF3131', width=3, dash='dash'), 
        name="AI預估軌跡"
    ), 1, 1)

    # 量能、MACD、KDJ (維持原樣)
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume']/1000, marker_color=v_colors), 2, 1)
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['MACD']-p_df['Signal'], marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in (p_df['MACD']-p_df['Signal'])]), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], line=dict(color='#00F5FF')), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], line=dict(color='#FFFF00')), 4, 1)

    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=False, paper_bgcolor='#000', plot_bgcolor='#000', margin=dict(l=10, r=10, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # --- [6-7] 底部 AI 診斷 HTML 盒 ---
    render_ai_diagnostic_box(insight, curr_p, stock_accuracy)

def render_ai_diagnostic_box(insight, curr_p, stock_accuracy):
    # 處理時間標籤
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tw_tz)
    
    # 預測下個交易日標籤
    next_day = now + timedelta(days=1)
    while next_day.weekday() >= 5: next_day += timedelta(days=1)
    
    pred_val = insight[3]
    est_color = "#FF3131" if pred_val > curr_p else "#00FF41"
    b_html = " | ".join([f"{k}D: <span style='color:{'#FF3131' if v >= 0 else '#00FF41'}'>{v:.2%}</span>" for k, v in insight[6].items()])

    html_content = f"""
    <div style="background-color: #0e1117; color: white; padding: 20px; border-radius: 12px; border: 1px solid #30363d; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div style="background: #FF3131; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold;">{stock_accuracy}</div>
            <div style="font-size: 24px; color: {insight[2]}; font-weight: 900;">{insight[0]}</div>
        </div>
        <hr style="border: 0; border-top: 1px solid #30363d; margin: 15px 0;">
        <p style="margin-bottom: 12px; font-size: 16px;"><b>AI 診斷分析：</b> {insight[1]}</p>
        <p style="font-size: 14px; color: #8b949e; margin-bottom: 20px;">當前乖離率參考：{b_html}</p>
        <div style="background-color: #161b22; padding: 18px; border-radius: 10px; border: 1px solid #30363d;">
            <div style="margin-bottom: 10px;">
                <div style="font-size: 14px; color: #8b949e;">預估 {next_day.strftime('%m/%d')} 收盤展望</div>
                <div style="font-size: 38px; color: {est_color}; font-weight: 900;">{pred_val:.2f}</div>
            </div>
            <div style="font-size: 15px; color: #c9d1d9;">
                壓力區間：<span style="color: #ff3131; font-weight: bold;">{insight[4]:.2f}</span> | 支撐區間：<span style="color: #00ff41; font-weight: bold;">{insight[5]:.2f}</span>
            </div>
        </div>
    </div>
    """
    components.html(html_content, height=400)
# =================================================================
# 第七章：主程式邏輯與權限控管 (2026 最終正確版 - 修復登入邏輯)
# =================================================================
import datetime as dt_module
import pytz
import time
import json
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

def main():
    # --- [7-1] Session 初始化與活動檢查 ---
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'last_active' not in st.session_state:
        st.session_state.last_active = time.time()
    
    # 未登入保護與 1 小時自動登出
    if st.session_state.user and (time.time() - st.session_state.last_active > 3600):
        st.session_state.clear()
        st.rerun()
    st.session_state.last_active = time.time()

    # --- [7-2] Google Sheets 連線引擎 ---
    @st.cache_resource(ttl=60)
    def get_gs_connection():
        try:
            if "gcp_service_account" in st.secrets:
                sc = st.secrets["gcp_service_account"]
            else:
                sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
            
            creds = Credentials.from_service_account_info(sc, scopes=[
                "https://www.googleapis.com/auth/spreadsheets", 
                "https://www.googleapis.com/auth/drive"
            ])
            target_url = st.secrets.get("spreadsheet_url") or st.secrets["connections"]["gsheets"]["spreadsheet"]
            sh_conn = gspread.authorize(creds).open_by_url(target_url)
            
            return {
                "users": sh_conn.worksheet("users"),
                "watchlist": sh_conn.worksheet("watchlist"),
                "settings": sh_conn.worksheet("settings"),
                "predictions": sh_conn.worksheet("predictions")
            }
        except Exception as e:
            st.error(f"📡 資料庫連線失敗: {e}")
            return None

    sheets = get_gs_connection()
    if not sheets: return
    ws_u, ws_w, ws_s, ws_p = sheets["users"], sheets["watchlist"], sheets["settings"], sheets["predictions"]

    # --- [7-3] 使用者身分驗證 UI (補零保險版) ---
    if st.session_state.user is None:
        st.markdown("""
            <style>
                [data-testid="stSidebar"] { display: none !important; }
                [data-testid="stSidebarNav"] { display: none !important; }
                .stMain { width: 100% !important; }
            </style>
        """, unsafe_allow_html=True)

        st.title("🚀 StockAI 台股決策終端")
        tab_login, tab_reg = st.tabs(["🔑 系統登入", "📝 註冊帳號"])
        
        try:
            user_data = ws_u.get_all_records()
            user_dict = {}
            for row in user_data:
                u = str(row['username']).strip()
                p = str(row['password']).strip()
                
                # 💡 [核心修正] 處理 Google Sheets 數字簡化問題
                # 如果密碼被簡化成 "0" 且你預期的是 "000000"，自動補齊
                if p == "0":
                    p = "000000"
                elif ".0" in p:
                    p = p.replace(".0", "")
                
                user_dict[u] = p
        except: 
            user_dict = {}

        with tab_login:
            u_name = st.text_input("帳號", key="login_u").strip()
            p_word = st.text_input("密碼", type="password", key="login_p").strip()
            
            if st.button("進入 AI 系統", use_container_width=True):
                input_p = str(p_word).strip()
                stored_p = user_dict.get(u_name)

                if stored_p:
                    # 💡 最終比對邏輯
                    if stored_p == input_p:
                        st.session_state.user = u_name
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ 密碼不符！(輸入長度: {len(input_p)}，資料庫轉換後長度: {len(stored_p)})")
                        st.info(f"系統目前的判定值為: {stored_p}")
                else:
                    st.error(f"❌ 找不到帳號 '{u_name}'")

        with tab_reg:
            st.warning("提醒：密碼請盡量包含英文字母，避免 Google Sheets 自動轉為數字格式。")
            new_u = st.text_input("設定新帳號", key="reg_u").strip()
            new_p = st.text_input("設定新密碼", type="password", key="reg_p").strip()
            if st.button("確認註冊", use_container_width=True):
                if new_u in user_dict: 
                    st.error("❌ 帳號已存在")
                elif new_u and new_p:
                    ws_u.append_row([str(new_u), str(new_p)])
                    st.success("🎉 註冊成功！請切換到登入頁籤。")
                    st.cache_data.clear()
        return

    # --- [7-4] 全域參數載入 ---
    try:
        s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
        cp = int(s_map.get('global_precision', 55))
        api_ttl = int(s_map.get('api_ttl_min', 1))
        tw_val = float(s_map.get('trend_weight', 1.0))
        v_comp = float(s_map.get('vol_comp', 1.5))
    except:
        cp, api_ttl, tw_val, v_comp = 55, 1, 1.0, 1.5

    # --- [7-5] 14:30 收盤自動化同步 (統一使用 symbol) ---
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = dt_module.datetime.now(tw_tz)
    
    # 💡 只有在開盤日的 14:30 後才觸發引擎
    if now_tw.time() >= dt_module.time(14, 30) and now_tw.weekday() < 5:
        with st.status("🌙 正在啟動收盤批次預測引擎...", expanded=False) as status:
            try:
                all_w_data = ws_w.get_all_records()
                if all_w_data:
                    # 💡 直接指定使用 'symbol' 欄位
                    unique_stocks = list(set([str(r['symbol']) for r in all_w_data if 'symbol' in r]))
                    
                    if unique_stocks:
                        run_batch_predict_engine(unique_stocks, ws_p, cp, tw_val, v_comp, api_ttl)
                        status.update(label=f"✅ 今日數據同步完成 (共 {len(unique_stocks)} 檔)", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 警告：試算表找不到 'symbol' 欄位標題", state="error")
            except Exception as e:
                st.error(f"⚠️ 同步異常: {e}")

    # --- [7-6] 管理面板：自選股維護 ---
    with st.expander("⚙️ 清單管理與系統設定", expanded=False):
        raw_w_data = ws_w.get_all_records()
        if raw_w_data:
            all_w_df = pd.DataFrame(raw_w_data)
            # 💡 統一指定欄位
            s_col = 'symbol' 
            u_col = 'username'
            u_stocks = all_w_df[all_w_df[u_col] == st.session_state.user][s_col].tolist() if u_col in all_w_df.columns else []
        else:
            u_stocks = []
            s_col = 'symbol'
            
        s_count = len(u_stocks)
        
        # 💡 [2026-01-15 需求實作] 20 支上限變色提醒邏輯
        s_color = "#FF3131" if s_count >= 20 else "#00F5FF"
        st.markdown(f"**目前自選股數量：** <span style='color:{s_color}; font-weight:bold; font-size:18px;'>{s_count} / 20</span>", unsafe_allow_html=True)
        
        if s_count >= 20:
            st.warning("🚨 提醒：您的自選股已達 20 支上限。若要新增標的，請先移除舊有項目。")

        col1, col2 = st.columns(2)
        with col1:
            # 防止清單為空時報錯，預設顯示台積電
            target_stock = st.selectbox("切換分析標的", u_stocks if u_stocks else ["2330.TW"])
            ns = st.text_input("➕ 新增股票 (例: 2454)")
            if st.button("確認加入"):
                if s_count >= 20:
                    st.error("🚫 操作攔截：已達 20 支上限，無法再加入。")
                elif ns:
                    raw_s = ns.upper().strip()
                    final_s = raw_s if "." in raw_s else (f"{raw_s}.TWO" if raw_s.startswith(('3','5','6','8')) else f"{raw_s}.TW")
                    if final_s not in u_stocks:
                        ws_w.append_row([st.session_state.user, final_s])
                        st.cache_data.clear() # 加入後強制清除快取刷新
                        st.rerun()
        with col2:
            p_days = st.number_input("AI 預測展望天數", 1, 30, 7)
            if st.button("🗑️ 移除此標的"):
                # 根據 username 與 symbol 進行刪除
                row = all_w_df[(all_w_df['username'] == st.session_state.user) & (all_w_df['symbol'] == target_stock)]
                if not row.empty:
                    ws_w.delete_rows(int(row.index[0]) + 2)
                    st.cache_data.clear()
                    st.rerun()
            if st.button("🚪 安全登出系統"):
                st.session_state.clear()
                st.rerun()

    # --- [7-7] 渲染介面 ---
    # 💡 呼叫渲染引擎，帶入剛剛選定的 target_stock
    render_terminal(target_stock, p_days, cp, tw_val, api_ttl, v_comp, ws_p)
