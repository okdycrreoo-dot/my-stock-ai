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

# --- [1-2 段] 全域背景與文字顏色設定 ---
# --- [1-3 段] 輸入框與下拉選單樣式 ---
# --- [1-4 段] 按鈕與摺疊面板樣式 ---
# --- [1-5 段] 診斷盒、AI建議盒與漲跌顏色標籤 ---
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

# --- [1-6 段] Google Sheets API 連線初始化 (穩定全域版) ---
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # 💡 優先讀取雲端 Secrets，若無則讀取本地 json
    if "connections" in st.secrets:
        # 配合您第七章的格式
        sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
        creds = Credentials.from_service_account_info(sc, scopes=scope)
    else:
        creds = Credentials.from_service_account_file("your_key.json", scopes=scope)
        
    client = gspread.authorize(creds)
    
    # 定義全域主物件
    sh = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
    ws_user = sh.worksheet("users")
    
except Exception as e:
    st.error(f"❌ [1-6 段] 初始化連線失敗：{e}")
    sh, ws_user = None, None
    
# =================================================================
# 第二章：數據引擎 (Data Engine)
# =================================================================

# --- [2-1 段] fetch_comprehensive_data 函數與 yfinance 下載邏輯 ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_data(symbol, ttl_seconds):
    raw_s = str(symbol).strip().upper()
    
    # 如果使用者已經手動輸入後綴，直接使用
    if raw_s.endswith(".TW") or raw_s.endswith(".TWO"):
        search_list = [raw_s]
    else:
        # 如果只輸入數字，優先嘗試上市 (.TW)，失敗則嘗試上櫃 (.TWO)
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        for _ in range(2):  # 每個後綴嘗試 2 次重試
            try:
                # 下載 2 年日線數據
                df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
                
                if df is not None and not df.empty and len(df) > 10:
                    # --- [2-2 段] 欄位處理 (MultiIndex 壓平) 與均線 (MA) 計算 ---
                    # 處理 yfinance 可能產生的多重索引欄位
                    if isinstance(df.columns, pd.MultiIndex): 
                        df.columns = df.columns.get_level_values(0)
                    
                    # 確保基礎欄位存在且為數值型別
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    
                    # 計算常用均線
                    df['MA5'] = df['Close'].rolling(5).mean()
                    df['MA10'] = df['Close'].rolling(10).mean()
                    df['MA20'] = df['Close'].rolling(20).mean()
                    
                    # --- [2-3 段] 技術指標計算 (MACD, KDJ, RSI, ATR) ---
                    # MACD 指標
                    e12 = df['Close'].ewm(span=12, adjust=False).mean()
                    e26 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = e12 - e26
                    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['Hist'] = df['MACD'] - df['Signal']
                    
                    # KDJ 隨機指標
                    l9 = df['Low'].rolling(9).min()
                    h9 = df['High'].rolling(9).max()
                    rsv = (df['Close'] - l9) / (h9 - l9 + 1e-9) * 100
                    df['K'] = rsv.ewm(com=2, adjust=False).mean()
                    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
                    df['J'] = 3 * df['K'] - 2 * df['D']
                    
                    # RSI 相對強弱指標
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
                    
                    # ATR 真實波幅均值
                    tr = pd.concat([
                        df['High'] - df['Low'], 
                        abs(df['High'] - df['Close'].shift()), 
                        abs(df['Low'] - df['Close'].shift())
                    ], axis=1).max(axis=1)
                    df['ATR'] = tr.rolling(14).mean()
                    
                    return df.dropna(), s
                
                time.sleep(1)
            except Exception as e:
                time.sleep(1)
                continue
    return None, raw_s
# =================================================================
# 第三章：自動對帳與反饋系統 (終極整合修正版)
# =================================================================

# --- [3-1 段] auto_sync_feedback 函數與時間判定邏輯 ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # 14:30 收盤判定邏輯
        is_after_market = (now.hour * 60 + now.minute) >= 870
        is_weekend = now.weekday() >= 5

# --- [3-2 段] 歷史對帳邏輯：以「次一交易日」收盤價進行精準對帳 ---
        if not df_p.empty:
            import time
            tw_tz = pytz.timezone('Asia/Taipei')
            # 取得台北今日日期 (1/16)
            today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")
            
            for i, row in df_p.iterrows():
                row_date = str(row.get('date', '')).strip()
                act_val = str(row.get('actual_close', '')).strip()
                err_val_str = str(row.get('error_pct', '')).strip()
                
                # 🚀 邏輯：只有日期「早於今天」的預測，今天才有收盤價可以對帳
                is_history = row_date < today_str
                needs_repair = is_history and (act_val == "" or act_val == "待收盤更新" or err_val_str == "")
                
                if needs_repair:
                    try:
                        # 🚀 關鍵偏移：1/15 預測 -> 抓取 1/16 的收盤價
                        check_date = (pd.to_datetime(row_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                        end_fetch = (pd.to_datetime(check_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        h = yf.download(row['symbol'], start=check_date, end=end_fetch, progress=False)
                        
                        if not h.empty:
                            act_df = h.copy()
                            if isinstance(act_df.columns, pd.MultiIndex):
                                act_df.columns = act_df.columns.get_level_values(0)
                            
                            # 取得 1/16 的 1360
                            actual_price = float(act_df['Close'].iloc[-1])
                            pred_price = float(row['pred_close'])
                            
                            # 計算誤差
                            err_val = (actual_price - pred_price) / (pred_price + 1e-9)
                            err_str = f"{err_val:.2%}"
                            
                            # 寫入 F 欄與 G 欄
                            ws_p.update_cell(i + 2, 6, round(actual_price, 2))
                            time.sleep(0.5) 
                            ws_p.update_cell(i + 2, 7, err_str)
                            
                            print(f"✅ {row['symbol']} 已用 {check_date} 收盤價補齊 {row_date} 的預測")
                    except Exception as e:
                        print(f"⚠️ {row['symbol']} 對帳失敗: {e}")
                        continue

# --- [3-3 段] UI 平均準確率計算 (🚀 已徹底移除區間命中率邏輯) ---
        if is_after_market and not is_weekend:
            # 🚀 這裡還原您原本的 next_bus_day 邏輯，但將寫入日期改為 today_str 以符合對帳邏輯
            next_bus_day = now + timedelta(days=1)
            while next_bus_day.weekday() >= 5: next_bus_day += timedelta(days=1)
            
            is_exists = any((str(r.get('date')) == today_str and r.get('symbol') == f_id) for r in recs)
            if not is_exists:
                # 寫入今日標籤 [1/15, symbol, 預測1/16價, ..., "待收盤更新", ""]
                new_row = [today_str, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "待收盤更新", ""]
                ws_p.append_row(new_row)
        
        # 計算 UI 顯示
        recs_latest = ws_p.get_all_records()
        df_latest = pd.DataFrame(recs_latest)
        df_stock = df_latest[(df_latest['symbol'] == f_id) & (df_latest['actual_close'] != "") & (df_latest['actual_close'] != "待收盤更新")].copy()
        
        accuracy_history = []
        avg_acc_text = "🎯 數據累積中"
        
        if not df_stock.empty:
            df_recent = df_stock.tail(10)
            total_acc = 0
            # 這裡就是您原本計算命中率的地方，我已將其簡化為純準確率計算
            for _, row in df_recent.iterrows():
                try:
                    act = float(row['actual_close'])
                    pred = float(row['pred_close'])
                    
                    # 🚀 僅保留準確率定義：1 - ABS(誤差率)
                    acc_val = (1 - abs(act - pred) / (pred + 1e-9)) * 100
                    acc_val = max(0, min(100, acc_val)) 
                    total_acc += acc_val
                    
                    accuracy_history.append({
                        "date": str(row['date'])[-5:], 
                        "acc_val": f"{acc_val:.1f}%",
                        "color": "#FF3131" if acc_val >= 98 else "#FFFFFF" 
                    })
                except: continue
            
            if len(accuracy_history) > 0:
                avg_acc = total_acc / len(accuracy_history)
                avg_acc_text = f"🎯 此股近期平均準確率: {avg_acc:.1f}%"
        
        return avg_acc_text, accuracy_history

    except Exception as e:
        return f"🎯 系統同步中...", []

# --- [3-4 段] 批次引擎：完整 AI 邏輯與 20 支限制提醒 ---
def run_batch_predict_engine(unique_stocks, ws_p, cp, tw_val, v_comp, api_ttl, ws_w):
    try:
        # 🚀 [新增自選股上限提醒]
        if len(unique_stocks) > 20:
            print(f"\n💡 【系統提醒】目前觀察名單共 {len(unique_stocks)} 支股票，已超過 20 支上限。")

        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        tw_tz = pytz.timezone('Asia/Taipei')
        today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")

        # 1. 補齊之前的「待收盤更新」 (同步修正偏移邏輯)
        if not df_p.empty:
            print("🔍 正在掃描是否存在未更新的歷史收盤價...")
            for i, row in df_p.iterrows():
                if str(row.get('actual_close', '')).strip() == "待收盤更新" and str(row.get('date', '')) < today_str:
                    try:
                        r_date = row['date']
                        # 🚀 偏移抓取：1/15 列補 1/16 價
                        check_date = (pd.to_datetime(r_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                        e_date = (pd.to_datetime(check_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        h = yf.download(row['symbol'], start=check_date, end=e_date, progress=False)
                        if not h.empty:
                            act_close = float(h['Close'].iloc[-1])
                            pred_close = float(row['pred_close'])
                            ws_p.update_cell(i + 2, 6, round(act_close, 2))
                            time.sleep(0.5)
                            err_val = (act_close - pred_close) / (pred_close + 1e-9)
                            ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")
                            print(f"✅ 已補齊 {row['symbol']} ({r_date}) 的次日收盤價")
                    except: continue

        # 2. 開始執行今日數據的預測與寫入 (保留所有原版運算細節)
        for symbol in unique_stocks:
            if not df_p.empty and 'symbol' in df_p.columns:
                is_done = not df_p[(df_p['symbol'] == symbol) & (df_p['date'] == today_str)].empty
                if is_done: continue
            
            try:
                # 這裡每一行都對應您提供的原版 AI 調用鏈
                df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
                if df is None: continue
                
                f_p, f_tw, f_v, _, bias, f_vol, b_drift = auto_fine_tune_engine(df, cp, tw_val, v_comp)
                _, _, _, _, _, _, _, insight = perform_ai_engine(df, 7, f_p, f_tw, f_v, bias, f_vol, b_drift)
                
                ws_p.append_row([
                    today_str, symbol, round(insight[3], 2), 
                    round(insight[5], 2), round(insight[4], 2), 
                    "待收盤更新", ""
                ])
                print(f"🚀 已完成 {symbol} 的今日預測寫入")
            except: continue
            
    except Exception as e:
        print(f"⚠️ 批次引擎執行異常: {e}")
# =================================================================
# 第四章：AI 微調引擎 (Fine-tune Engine)
# =================================================================

# --- [4-1 段] auto_fine_tune_engine 函數與大盤判斷 ---
def auto_fine_tune_engine(df, base_p, base_tw, v_comp):
    try:
        mkt_df = yf.download("^TWII", period="1mo", interval="1d", auto_adjust=True, progress=False)
        mkt_rets = mkt_df['Close'].pct_change().dropna()
        mkt_vol = mkt_rets.tail(20).std()
        env_panic = 1.25 if mkt_vol > 0.012 else 1.0
    except:
        env_panic = 1.0

    # --- [4-2 段] 波動率與趨勢權重的多維度計算 ---
    rets = df['Close'].pct_change().dropna()
    v_p = [5, 10, 15, 20, 25, 30]
    v_w = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    v_vals = [rets.tail(p).std() for p in v_p]
    
    f_vol = sum(v * w for v, w in zip(v_vals, v_w)) * env_panic
    
    v_curr = df['Volume'].iloc[-1]
    v_avg5 = df['Volume'].tail(5).mean()
    vol_ratio = v_curr / (v_avg5 + 0.1)
    
    tw_adj = 0.8 if env_panic > 1.0 else 1.0
    f_tw = max(0.5, min(2.5, 1.0 + (rets.tail(5).mean() * 15 * min(1.5, vol_ratio)) * tw_adj))
    
    # --- [4-3 段] 乖離率偏好、標本群漂移與 AI 推薦參數生成 ---
    price_now = float(df['Close'].iloc[-1])
    b_periods = [5, 10, 15, 20, 25, 30]
    b_weights = [0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
    bias_list = []
    for p in b_periods:
        ma_tmp = df['Close'].rolling(p).mean().iloc[-1]
        bias_list.append((price_now - ma_tmp) / (ma_tmp + 1e-5))
    bias_val = sum(b * w for b, w in zip(bias_list, b_weights))
    
    f_p = (45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60)
    if env_panic > 1.0: f_p = int(f_p * 0.85)

    high_low_range = (df['High'] - df['Low']).tail(5).mean() / price_now
    f_v = 1.3 if high_low_range > 0.035 else 2.1 if high_low_range < 0.015 else 1.7
    
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

# --- [5-1 段] perform_ai_engine 變數初始化與主力力道矩陣 ---
def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol, b_drift):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    sens = (int(precision) / 55)
    curr_p = float(last['Close'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100

    v_avg20 = df['Volume'].tail(20).mean() 
    vol_ratio = curr_v / (v_avg20 + 0.1)

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

    # --- [5-2 段] 進階指標 A-C (布林擠壓、乖離力竭、多空排列) ---
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

    std_20 = df['Close'].rolling(20).std()
    bb_width = (std_20 * 4) / (df['MA20'] + 1e-5) 
    is_squeezing = bb_width.iloc[-1] < bb_width.tail(20).mean() * 0.92
    squeeze_boost = 1.35 if is_squeezing else 1.0

    curr_bias = (curr_p - last['MA20']) / (last['MA20'] + 1e-5)
    prev_bias = (prev_c - prev['MA20']) / (prev['MA20'] + 1e-5)
    exhaustion_drag = -0.0018 if (curr_p > prev_c and curr_bias < prev_bias) else 0

    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    ma_perfect_order = 1.25 if (last['MA5'] > last['MA10'] > last['MA20'] > ma60) else 1.0

    # --- [5-3 段] 進階指標 D-G (均線斜率、ATR-Bias、量價背離、波動壓縮) ---
    ma10_s = df['MA10'].diff(3) 
    slope_now = ma10_s.iloc[-1]
    slope_prev = ma10_s.iloc[-3]
    slope_decay = -0.0015 if (slope_now > 0 and slope_now < slope_prev) else 0

    atr_val = last['ATR']
    dist_from_ma20 = curr_p - last['MA20']
    normalized_bias = dist_from_ma20 / (atr_val + 1e-5)
    vol_bias_pull = -0.002 if normalized_bias > 2.0 else 0.002 if normalized_bias < -2.0 else 0

    vp_divergence = -0.0025 if (change_pct > 0.5 and vol_ratio < 0.8) else 0

    atr_long_avg = df['ATR'].tail(60).mean()
    vol_gap_boost = 1.4 if (last['ATR'] < atr_long_avg * 0.75) else 1.0

    # --- [5-4 段] 進階指標 H-K (資金流 MFI、乖離加速度、蔡金波動衰竭、RSI 動能) ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    flow_dir = np.where(tp > tp.shift(1), 1, -1)
    pos_mf = rmf.where(flow_dir > 0, 0).tail(14).sum()
    neg_mf = rmf.where(flow_dir < 0, 0).tail(14).sum()
    mfi_val = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-5))))
    mfi_drag = -0.0035 if (change_pct > 0.3 and mfi_val < 45) else 0

    bias_v = normalized_bias - (prev_c - prev['MA20'])/(prev['ATR']+1e-5)
    bias_accel = 0.0015 if (normalized_bias > 1.2 and bias_v > 0) else 0

    hl_ema = (df['High'] - df['Low']).ewm(span=10).mean()
    chv = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-5)
    vol_exhaustion = -0.003 if (chv.iloc[-1] < -0.2 and change_pct > 0.5) else 0.002 if (chv.iloc[-1] < -0.2 and change_pct < -0.5) else 0

    rsi_s = df['RSI'].diff(3).iloc[-1]
    rsi_mom_boost = 0.0025 if (last['RSI'] > 50 and rsi_s > 5) else -0.0025 if (last['RSI'] < 50 and rsi_s < -5) else 0

    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    # --- [5-5 段] 蒙地卡羅模擬運算邏輯 ---
    np.random.seed(42)
    sim_results = []
    
    base_drift = (((int(precision) - 55) / 1000) * float(trend_weight) * ma_perfect_order + 
                  (rsi_div * 0.0025) + (chip_mom * 0.15) + (b_drift * 0.22) + 
                  exhaustion_drag + slope_decay + vol_bias_pull + vp_divergence + 
                  mfi_drag + bias_accel + vol_exhaustion + rsi_mom_boost + 
                  whale_force + whale_dump)
    
    for _ in range(1000):
        noise = np.random.normal(0, f_vol * v_comp * vol_contract * squeeze_boost * vol_gap_boost, p_days)
        path = [curr_p]
        for i in range(p_days):
            reversion_pull = bias * 0.08
            next_p = path[-1] * (1 + base_drift - reversion_pull + noise[i])
            path.append(next_p)
        sim_results.append(path[1:])
    
    pred_prices = np.mean(sim_results, axis=0)
    next_close = pred_prices[0]
    std_val = np.std([p[0] for p in sim_results])
    
    # --- [5-6 段] 診斷建議與多空評分系統 ---
    # (此段接續 5-5 段的計算結果)
    ma_check_list = [5, 10, 15, 20, 25, 30]
    above_ma_count = sum(1 for p in ma_check_list if curr_p > df['Close'].rolling(p).mean().iloc[-1])

    score = 0
    reasons = []
    
    # --- 1. 動態指標特徵判定 ---
    if ma_perfect_order > 1.0: 
        score += 2; reasons.append("多頭完美排列(飆股模式)")
    elif above_ma_count >= 5: 
        score += 1.5; reasons.append(f"均線多頭排列")
    
    if is_squeezing: reasons.append("布林極度擠壓(即將噴發)")
    if exhaustion_drag < 0: score -= 0.5; reasons.append("漲勢背離力竭")
    
    if slope_decay < 0: score -= 0.3; reasons.append("均線慣性減速")
    
    if normalized_bias > 2.0: 
        score -= 0.5; reasons.append("波動超漲(引力修正)")
    elif normalized_bias < -2.0: 
        score += 0.5; reasons.append("波動超跌(引力支撐)")
    
    if vp_divergence < 0 or mfi_drag < 0:
        score -= 0.5; reasons.append("量價資金背離(警惕虛漲)")
    if mfi_val > 80:
        score -= 0.2; reasons.append("資金極度過熱")
    if bias_accel > 0:
        score += 0.4; reasons.append("乖離加速度(強勢主升段)")
    if vol_exhaustion < 0:
        score -= 0.4; reasons.append("波動率力竭(漲勢過激)")
    if rsi_mom_boost > 0:
        reasons.append("RSI動能爆發")
    if vol_gap_boost > 1.0:
        reasons.append("波動率極度壓縮(變盤在即)")

    if whale_force > 0: score += 1.2; reasons.append("偵測大戶敲單進場")
    if whale_dump < 0: score -= 1.2; reasons.append("大戶棄守逃命跡象")
    if change_pct > 1.2 and vol_ratio > 1.3: score += 1; reasons.append("法人級放量攻擊")
    if b_drift > 0.003: score += 1; reasons.append("標本群體向上共振")
    
    # --- 2. [新增] 保底邏輯：若無明顯異動特徵，則給予狀態描述 ---
    if not reasons:
        if score >= 1:
            reasons.append("走勢溫和偏多，建議沿均線擇優布局")
        elif score <= -1:
            reasons.append("走勢疲軟偏弱，建議持股汰弱留強")
        else:
            reasons.append("目前處於箱型整理區間，建議觀望靜待量能突破")

    # --- 3. 狀態映射 ---
    status_map = {
        3: ("🚀 強力買入", "#FF3131"), 
        2: ("🚀 強力買入", "#FF3131"), 
        1: ("📈 偏多操作", "#FF7A7A"), 
        0: ("⚖️ 觀望中性", "#FFFF00"), 
        -1: ("📉 偏空警戒", "#00FF41"), 
        -2: ("📉 偏空警戒", "#00FF41")
    }
    res = status_map.get(max(-2, min(3, int(score))), ("⚖️ 觀望中性", "#FFFF00"))
    
    # --- 4. 買賣建議區間計算 ---
    adv = {
        k: {
            "buy": m * (1 - f_vol * v_comp * f * sens), 
            "sell": m * (1 + f_vol * v_comp * f * sens)
        } for k, (m, f) in {
            "5日極短線建議": (df['Close'].rolling(5).mean().iloc[-1], 0.8), 
            "10日短線建議": (df['Close'].rolling(10).mean().iloc[-1], 1.1), 
            "20日波段建議": (last['MA20'], 1.5)
        }.items()
    }
    
    # --- 5. 乖離率數據準備 ---
    b_sum = {p: (curr_p - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5) for p in [5, 10, 20, 30]}
    
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), b_sum)

# =================================================================
# 第六章：終端渲染引擎 (Render Terminal)
# =================================================================
from datetime import datetime, timedelta  # 💡 移到最上方，修復 image_f82fe5 的報錯
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- [6-1 段] render_terminal 完整呼叫邏輯 ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p):
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗"); return

    final_p, final_tw, ai_v, ai_b, bias, f_vol, b_drift = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(
        df, p_days, final_p, final_tw, ai_v, bias, f_vol, b_drift
    )
    
    # 重點：這裡必須同時接收文字(stock_accuracy)與清單(acc_history)
    stock_accuracy, acc_history = auto_sync_feedback(ws_p, f_id, insight)

    st.markdown("""
        <style>
        .stApp { background-color: #000000; }
        .streamlit-expanderHeader { background-color: #FF3131 !important; color: white !important; font-weight: 900 !important; }
        .info-box { background: #0A0A0A; padding: 12px; border: 1px solid #333; border-radius: 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100px; }
        .diag-box { background: #050505; padding: 15px; border-radius: 12px; border: 1px solid #444; min-height: 120px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .ai-advice-box { background: #000000; border: 2px solid #333; padding: 20px; border-radius: 15px; margin-top: 25px; }
        .confidence-tag { background: #FF3131; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

# --- [6-2 段] 頂部核心指標看板與 10 日精確準確率紀錄 ---
    # 渲染大標題
    st.title(f"📊 {f_id} 台股AI輔助系統")
    
    # 渲染橫向 10 日準確率數值紀錄
    if acc_history:
        acc_cols = st.columns(len(acc_history))
        for i, item in enumerate(acc_history):
            with acc_cols[i]:
                st.markdown(f"""
                    <div style='text-align: center; border: 1px solid #333; border-radius: 8px; padding: 5px; background: #111; margin-bottom: 10px;'>
                        <div style='font-size: 0.7rem; color: #888; font-weight: bold;'>{item['date']}</div>
                        <div style='font-size: 0.9rem; margin-top: 2px; color: {item['color']}; font-weight: 900;'>{item['acc_val']}</div>
                    </div>
                """, unsafe_allow_html=True)

    # 顯示整體命中率標籤
    st.markdown(f"<div class='confidence-tag'>{stock_accuracy}</div>", unsafe_allow_html=True)
    st.caption(f"✨ AI 大腦：籌碼與動能分析 | 環境共振分析 | 技術面與乖離率評估 | 自我學習與反饋")

    # 核心指標看板佈局 (Metrics)
    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""
    m_cols = st.columns(5)
    metrics = [
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), 
        ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), 
        ("當前價格", f"{curr_p:.2f}", c_p), 
        ("今日漲跌", f"{sign}{change_pct:.2f}%", c_p), 
        ("成交 (張)", f"{int(curr_v/1000):,}", "#FFFF00")
    ]
    
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: 
            st.markdown(f"""
                <div class='info-box'>
                    <span style='color:#888; font-size:1.1rem; margin-bottom:5px;'>{lab}</span>
                    <b style='color:{col}; font-size:2.0rem; line-height:1;'>{val}</b>
                </div>
            """, unsafe_allow_html=True)

# --- [6-3 段] 極短線/短線/波段買賣點診斷區 ---
    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: 
            st.markdown(f"<div class='diag-box'><b style='font-size:1.5rem; color:#FFFFFF;'>{label}</b><hr style='border:0.5px solid #444; width:80%; margin:10px 0;'><div style='font-size:1.2rem; color:#CCC;'>買入: <span style='color:#FF3131; font-weight:900; font-size:1.6rem;'>{p['buy']:.2f}</span></div><div style='font-size:1.2rem; color:#CCC;'>賣出: <span style='color:#00FF41; font-weight:900; font-size:1.6rem;'>{p['sell']:.2f}</span></div></div>", unsafe_allow_html=True)

# --- [6-4 段] Plotly 四層子圖繪製 (K線、量能、MACD、KDJ) ---
    t_main = "■ 價格與均線 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#FF3131'>●</span> K線 <span style='color:#FFD700'>━━</span> 5MA <span style='color:#00F5FF'>━━</span> 10MA <span style='color:#FF00FF'>━━</span> 20MA <span style='color:#FF3131'>···</span> AI預測</span>"
    t_vol  = "■ 成交量分析 (張)"
    t_macd = "■ MACD 指標 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#FF3131'>■</span> 能量柱 <span style='color:#FFFFFF'>━━</span> DIF <span style='color:#FFA726'>━━</span> DEA</span>"
    t_kdj  = "■ KDJ 擺動指標 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#00F5FF'>━━</span> K值 <span style='color:#FFFF00'>━━</span> D值 <span style='color:#E066FF'>━━</span> J值</span>"

    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.4, 0.15, 0.2, 0.25], 
        vertical_spacing=0.04, 
        subplot_titles=(t_main, t_vol, t_macd, t_kdj)
    )
    p_df = df.tail(90)
    
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], line=dict(color='#FFD700', width=2), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA10'], line=dict(color='#00F5FF', width=1.5), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], line=dict(color='#FF00FF', width=2), showlegend=False), 1, 1)
    
    # 預測線繪製 (確保 pred_line 已傳入)
    fig.add_trace(go.Scatter(x=[p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)], y=pred_line, line=dict(color='#FF3131', width=3, dash='dash'), showlegend=False), 1, 1)
    
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume']/1000, marker_color=v_colors, showlegend=False), 2, 1)
    
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']], showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], line=dict(color='#FFFFFF', width=1.2), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], line=dict(color='#FFA726', width=1.2), showlegend=False), 3, 1)
    
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], line=dict(color='#00F5FF'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], line=dict(color='#FFFF00'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], line=dict(color='#E066FF'), showlegend=False), 4, 1)

    fig.update_layout(template="plotly_dark", height=880, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=50, b=50), paper_bgcolor='#000000', plot_bgcolor='#000000')
    
    for i in fig['layout']['annotations']:
        i['x'] = 0; i['xanchor'] = 'left'; i['font'] = dict(size=14, color="#FFFFFF")

    st.plotly_chart(fig, use_container_width=True)

   # --- [6-5 段] 底部 AI 診斷建議盒 (視覺顏色對應修正版) ---
    # 💡 注意：此處已移除重複導入，以確保 6-4 段的技術圖表不崩潰
    
    now = datetime.now()
    today_label = now.strftime("%m/%d")
    
    # 計算預計交易日標籤
    next_day = now + timedelta(days=1)
    while next_day.weekday() >= 5: next_day += timedelta(days=1)
    next_day_label = next_day.strftime("%m/%d")

    # 💡 判斷預估價顏色：高於現價用紅(#FF3131)，低於用綠(#00F5FF)
    pred_val = insight[3]
    est_color = "#FF3131" if pred_val > curr_p else "#00F5FF"

    b_html = " | ".join([f"{k}D: <span style='color:{'#FF3131' if v >= 0 else '#00F5FF'}'>{v:.2%}</span>" for k, v in insight[6].items()])
    acc_val_display = stock_accuracy.split(':')[-1].strip() if '命中率' in stock_accuracy else "計算中"

    html_content = f"""
    <div style="background-color: #0e1117; color: white; padding: 20px; border-radius: 12px; border: 1px solid #30363d; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div style="background: #FF4B4B; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold;">{stock_accuracy}</div>
            <div style="font-size: 24px; color: {insight[2]}; font-weight: 900;">{insight[0]}</div>
        </div>
        <hr style="border: 0; border-top: 1px solid #30363d; margin: 15px 0;">
        <p style="margin-bottom: 12px; font-size: 16px;"><b>AI 診斷建議：</b> {insight[1]}</p>
        <p style="font-size: 14px; color: #8b949e; margin-bottom: 20px;">當前 {today_label} 乖離率參考：{b_html}</p>
        <div style="background-color: #161b22; padding: 18px; border-radius: 10px; border: 1px solid #30363d;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="color: #58a6ff; font-weight: bold; font-size: 16px;">🔮 AI 統一展望 ({today_label})</span>
                <span style="color: #3fb950; font-size: 12px; border: 1px solid #30363d; padding: 2px 8px; border-radius: 5px;">命中率: {acc_val_display}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="font-size: 14px; color: #8b949e;">預估 {next_day_label} 收盤價</div>
                <div style="font-size: 38px; color: {est_color}; font-weight: 900;">{pred_val:.2f}</div>
            </div>
            <div style="font-size: 15px; color: #c9d1d9;">
                預估價格區間：<span style="color: #ff7b72; font-weight: bold;">{insight[5]:.2f}</span> ~ <span style="color: #ff7b72; font-weight: bold;">{insight[4]:.2f}</span>
            </div>
        </div>
    </div>
    """
    components.html(html_content, height=450)
# =================================================================
# 第七章：主程式邏輯與權限控管 (正式對齊版 - 2026-01-16)
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
    # -------------------------------------------------------------
    # [段落 7-1] Session 初始化與活動檢查
    # -------------------------------------------------------------
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'last_active' not in st.session_state:
        st.session_state.last_active = time.time()
    
    # 未登入時清空快取以防 API 報錯
    if st.session_state.user is None:
        st.cache_data.clear()
    
    # 1小時自動登出邏輯
    if st.session_state.user and (time.time() - st.session_state.last_active > 3600):
        st.session_state.clear()
        st.rerun()
    st.session_state.last_active = time.time()

    # -------------------------------------------------------------
    # [段落 7-2] 使用者身分驗證 UI
    # -------------------------------------------------------------
    if st.session_state.user is None:
        st.title("🚀 StockAI 台股AI分析系統")
        tab_login, tab_reg = st.tabs(["🔑 系統登入", "📝 註冊帳號"])
        
        current_ws = globals().get('ws_user')
        user_dict = {}
        if current_ws:
            try:
                user_data = current_ws.get_all_records()
                user_dict = {str(row['username']): str(row['password']) for row in user_data}
            except: pass

        with tab_login:
            u_name = st.text_input("帳號", key="login_u").strip()
            p_word = st.text_input("密碼", type="password", key="login_p").strip()
            if st.button("立即進入系統", use_container_width=True):
                if u_name in user_dict and str(user_dict[u_name]) == p_word:
                    st.session_state.user = u_name
                    st.rerun()
                else:
                    st.error("❌ 帳號或密碼錯誤")

        with tab_reg:
            st.subheader("建立新帳戶")
            new_u = st.text_input("設定新帳號", key="reg_u").strip()
            new_p = st.text_input("設定新密碼", type="password", key="reg_p").strip()
            if st.button("確認註冊並寫入系統", use_container_width=True):
                if not current_ws:
                    st.error("❌ 系統連線未建立")
                elif new_u in user_dict:
                    st.error(f"❌ 帳號 '{new_u}' 已存在")
                elif new_u and new_p:
                    current_ws.append_row([new_u, new_p])
                    st.balloons()
                    st.success(f"🎉 註冊成功！請登入。")
                else:
                    st.warning("帳號密碼不可為空")
        return 

    # -------------------------------------------------------------
    # [段落 7-3] 資料庫連線與全局參數
    # -------------------------------------------------------------
    @st.cache_resource(ttl=30)
    def get_gs_connection():
        try:
            if "gcp_service_account" in st.secrets:
                sc = st.secrets["gcp_service_account"]
            else:
                sc = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
            
            creds = Credentials.from_service_account_info(sc, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
            target_url = st.secrets.get("spreadsheet_url") or st.secrets["connections"]["gsheets"]["spreadsheet"]
            sh_conn = gspread.authorize(creds).open_by_url(target_url)
            
            return {
                "users": sh_conn.worksheet("users"),
                "watchlist": sh_conn.worksheet("watchlist"),
                "settings": sh_conn.worksheet("settings"),
                "predictions": sh_conn.worksheet("predictions")
            }
        except: return None

    sheets = get_gs_connection()
    if not sheets: return
    ws_u, ws_w, ws_s, ws_p = sheets["users"], sheets["watchlist"], sheets["settings"], sheets["predictions"]

    try:
        s_map = {r['setting_name']: r['value'] for r in ws_s.get_all_records()}
        cp, api_ttl, tw_val, v_comp = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 1)), float(s_map.get('trend_weight', 1.0)), float(s_map.get('vol_comp', 1.5))
    except:
        cp, api_ttl, tw_val, v_comp = 55, 1, 1.0, 1.5

    # -------------------------------------------------------------
# [第七章 修正版 7-4 段] 批次引擎：14:30 收盤門禁 + 跨使用者去重同步
# -------------------------------------------------------------
    # 設定台灣時區與檢查目前時間
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = dt_module.datetime.now(tw_tz)
    market_close = dt_module.time(14, 30)

    # 判斷是否進入收盤後批次處理時間
    if now_tw.time() >= market_close:
        with st.spinner("🌙 偵測到收盤時間，正在執行全清單 AI 預測同步..."):
            try:
                # 1. 從 Google Sheets 抓取所有使用者的自選股
                all_w_data = ws_w.get_all_records()
                
                if all_w_data:
                    # 2. 跨使用者去重邏輯：提取所有不重複的股票代號
                    # 💡 即便多人追蹤同一檔，今日也只會計算並寫入一次 predictions 表
                    unique_stocks = list(set([str(r['stock_symbol']) for r in all_w_data]))
                    
                    # 3. 呼叫批次引擎 (關鍵修正：補上 ws_w 參數確保內部可讀取清單)
                    run_batch_predict_engine(
                        unique_stocks, # 待測清單
                        ws_p,          # 預測表連線
                        cp,            # 全域精度
                        tw_val,        # 趨勢權重
                        v_comp,        # 波動補償
                        api_ttl,       # API 快取時間
                        ws_w           # 自選股表連線 (💡 修復錯誤的關鍵)
                    )
                    st.success(f"✅ 已完成 {len(unique_stocks)} 檔標的之收盤預測寫入。")
                else:
                    st.warning("目前無任何使用者加入自選股，跳過批次引擎。")
                    
            except Exception as e:
                st.error(f"❌ 收盤同步執行異常: {e}")
    else:
        # 盤中模式僅提示
        st.info(f"☀️ 盤中時間 ({now_tw.strftime('%H:%M')}) 僅提供即時分析，14:30 後將產出正式預測報告。")

    # -------------------------------------------------------------
    # [段落 7-5] 管理面板：自選股維護 (含 20 支上限邏輯)
    # -------------------------------------------------------------
    with st.expander("⚙️ 管理自選股清單", expanded=False):
        all_w_df = pd.DataFrame(ws_w.get_all_records())
        u_stocks = all_w_df[all_w_df['username'] == st.session_state.user]['stock_symbol'].tolist() if not all_w_df.empty else []
        s_count = len(u_stocks)
        
        m1, m2 = st.columns(2)
        with m1:
            # 💡 [2026-01-15] 實作：上限 20 支變色提醒
            s_color = "#FF3131" if s_count >= 20 else "#00F5FF"
            st.markdown(f"### 自選股清單上限：<span style='color:{s_color}'>{s_count} / 20</span>", unsafe_allow_html=True)
            target = st.selectbox("選取分析標的", u_stocks if u_stocks else ["2330.TW"])
            ns = st.text_input("➕ 新增代號")
            if st.button("確認加入追蹤"):
                if s_count >= 20:
                    st.error("🚫 提醒：自選股已達 20 支上限！")
                elif ns:
                    raw_s = ns.upper().strip()
                    final_s = raw_s if "." in raw_s else (f"{raw_s}.TWO" if raw_s.startswith(('3','5','6','8')) else f"{raw_s}.TW")
                    if final_s not in u_stocks:
                        ws_w.append_row([st.session_state.user, final_s])
                        st.rerun()
        with m2:
            p_days = st.number_input("AI 預估天數", 1, 30, 7)
            if st.button("🗑️ 刪除目前標的"):
                row = all_w_df[(all_w_df['username'] == st.session_state.user) & (all_w_df['stock_symbol'] == target)]
                if not row.empty:
                    ws_w.delete_rows(int(row.index[0]) + 2)
                    st.rerun()
            if st.button("🚪 安全登出系統"):
                st.session_state.clear()
                st.rerun()

    # -------------------------------------------------------------
    # [段落 7-6] 核心運算對接與介面渲染 (恢復 AI 診斷輸出)
    # -------------------------------------------------------------
    df, f_id = fetch_comprehensive_data(target, api_ttl * 60)
    if df is not None:
        f_p, f_tw, f_v, _, bias, f_vol, b_drift = auto_fine_tune_engine(df, p_days, tw_val, v_comp)
        curr_p, open_p, last_p, change, curr_v, ma_vals, acc_cols, insight = perform_ai_engine(
            df, p_days, f_p, f_tw, f_v, bias, f_vol, b_drift
        )
        # 💡 同步反饋與診斷介面渲染
        auto_sync_feedback(ws_p, f_id, insight)
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)
    else:
        st.error("數據獲取異常，請檢查代號。")

if __name__ == "__main__":
    main()















