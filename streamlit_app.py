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
def fetch_comprehensive_data(symbol, ttl_seconds, refresh_key):
    # refresh_key 只是用來打破緩存，函數內不需要用到它
    s = str(symbol).strip().upper()
    if not (s.endswith(".TW") or s.endswith(".TWO")): 
        s = f"{s}.TW"
    for _ in range(3):
        try:
            # 1. 下載歷史序列
            df = yf.download(s, period="2y", interval="1d", progress=False, ignore_tz=True)
            
            # 2. 強制獲取即時快照 (解決 13:30 結算後歷史數據未更新的問題)
            tk = yf.Ticker(s)
            try:
                # 獲取最新成交資訊 (fast_info 通常比 history 快)
                info = tk.fast_info
                last_price = info['last_price']
                last_time = info['last_evaluation'].date()
                
                # 檢查：如果歷史數據的最後一天早於即時數據的日期
                if df.index[-1].date() < last_time:
                    # 建立今日的補丁 DataFrame
                    patch_row = pd.DataFrame({
                        'Open': [info['open']],
                        'High': [info['day_high']],
                        'Low': [info['day_low']],
                        'Close': [last_price],
                        'Volume': [info['last_volume']]
                    }, index=[pd.to_datetime(last_time)])
                    
                    df = pd.concat([df, patch_row])
                    df = df[~df.index.duplicated(keep='last')] # 確保不重複
            except:
                pass # 若快照獲取失敗，維持原歷史序列
            
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

# --- 3. 背景自動對帳與全清單權威更新 (唯一完整修正版) ---
def auto_sync_feedback(ws_p, ws_w, f_id, insight, cp, tw_val, v_comp, p_days, api_ttl):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        watchlist = pd.DataFrame(ws_w.get_all_records())
        unique_stocks = watchlist['stock_symbol'].unique().tolist()
        
        today = datetime.now().strftime("%Y-%m-%d")
        r_key = datetime.now().strftime("%Y-%m-%d %H:%M") 
        is_weekend = datetime.now().weekday() >= 5
        now = datetime.now()
        is_finalized = (now.hour > 14) or (now.hour == 14 and now.minute >= 30)

        if not is_weekend:
            # --- A. 自動對帳逻辑 ---
            for i, row in df_p.iterrows():
                if str(row['actual_close']) == "" and row['date'] != today:
                    try:
                        h = yf.download(row['symbol'], start=row['date'], end=(pd.to_datetime(row['date']) + timedelta(days=3)).strftime("%Y-%m-%d"), progress=False)
                        if not h.empty:
                            act_close = float(h['Close'].iloc[0])
                            ws_p.update_cell(i + 2, 6, round(act_close, 2))
                    except: continue

            # --- B. 全清單寫入 (3017 未寫入的關鍵修復) ---
            if is_finalized:
                for stock in unique_stocks:
                    existing = df_p[(df_p['date'] == today) & (df_p['symbol'] == stock)]
                    if stock == f_id:
                        p_val = round(insight[3], 2)
                        if existing.empty:
                            ws_p.append_row([today, stock, p_val, round(insight[5], 2), round(insight[4], 2), "", ""])
                        else:
                            # 即使已有資料，若數值不對也會強制更新
                            row_idx = existing.index[0] + 2
                            if abs(float(existing.iloc[0]['pred_close']) - p_val) > 0.01:
                                ws_p.update_cell(row_idx, 3, p_val)
                    elif existing.empty:
                        # 靜默更新其他標的
                        try:
                            tmp_df, _ = fetch_comprehensive_data(stock, api_ttl * 60, r_key)
                            if tmp_df is not None:
                                f_p, f_tw, ai_v, ai_b, bias, f_vol, b_drift = auto_fine_tune_engine(tmp_df, cp, tw_val, v_comp)
                                _, _, _, _, _, _, _, tmp_i = perform_ai_engine(tmp_df, p_days, f_p, f_tw, ai_v, bias, f_vol, b_drift)
                                ws_p.append_row([today, stock, round(tmp_i[3], 2), round(tmp_i[5], 2), round(tmp_i[4], 2), "", ""])
                        except: continue

        return None # 正常結束
    except Exception as e:
        print(f"Sync Error: {e}")
        return None
        
# --- 4. AI 核心：深度微調連動引擎 (進階指標增強版) ---
def auto_fine_tune_engine(df, base_p, base_tw, v_comp):
    try:
        mkt_df = yf.download("^TWII", period="1mo", interval="1d", auto_adjust=True, progress=False)
        mkt_rets = mkt_df['Close'].pct_change().dropna()
        mkt_vol = mkt_rets.tail(20).std()
        env_panic = 1.25 if mkt_vol > 0.012 else 1.0
    except:
        env_panic = 1.0

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

# --- 5. 預測運算引擎 (核心公式注入層) ---
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

    # --- [新增指標 L] 主力力道矩陣 (Whale Force Matrix) ---
    # 當 漲幅 > 2% 且 量增 > 50% 時，定義為主力表態攻擊
    whale_force = (change_pct * 0.002) if (change_pct > 2.0 and vol_ratio > 1.5) else 0
    # 若 跌幅 > 2% 且 量增 > 50% 時，定義為主力棄守逃命
    whale_dump = (change_pct * 0.0015) if (change_pct < -2.0 and vol_ratio > 1.5) else 0

    # 籌碼動能判斷
    if change_pct > 0.5 and vol_ratio > 1.2:
        chip_mom = (change_pct / 100) * vol_ratio * 1.5 
    elif change_pct < 0 and vol_ratio < 0.7:
        chip_mom = abs(change_pct / 100) * 0.2 
    elif change_pct < -1.5 and vol_ratio > 1.5:
        chip_mom = (change_pct / 100) * vol_ratio * 1.2
    else:
        chip_mom = (change_pct / 100)

    # RSI 背離偵測
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

    # --- [核心指標增強 A] 布林通道擠壓偵測 ---
    std_20 = df['Close'].rolling(20).std()
    bb_width = (std_20 * 4) / (df['MA20'] + 1e-5) # 移除 .iloc[-1]
    is_squeezing = bb_width.iloc[-1] < bb_width.tail(20).mean() * 0.92
    squeeze_boost = 1.35 if is_squeezing else 1.0

    # --- [核心指標增強 B] 乖離力竭偵測 ---
    curr_bias = (curr_p - last['MA20']) / (last['MA20'] + 1e-5)
    prev_bias = (prev_c - prev['MA20']) / (prev['MA20'] + 1e-5)
    exhaustion_drag = -0.0018 if (curr_p > prev_c and curr_bias < prev_bias) else 0

    # --- [核心指標增強 C] 多空全排列強度 ---
    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    ma_perfect_order = 1.25 if (last['MA5'] > last['MA10'] > last['MA20'] > ma60) else 1.0

    # --- [新增指標 D] 均線斜率變動率 (Slope Decay) ---
    # 計算 MA10 的變動斜率：若斜率從正轉平，代表動能衰減
    ma10_s = df['MA10'].diff(3) # 觀察 3 天內的 MA10 位移
    slope_now = ma10_s.iloc[-1]
    slope_prev = ma10_s.iloc[-3]
    # 如果還在漲但斜率變小，給予負向阻力
    slope_decay = -0.0015 if (slope_now > 0 and slope_now < slope_prev) else 0

    # --- [新增指標 E] 波動校正乖離 (ATR-Bias) ---
    atr_val = last['ATR']
    dist_from_ma20 = curr_p - last['MA20']
    normalized_bias = dist_from_ma20 / (atr_val + 1e-5)
    vol_bias_pull = -0.002 if normalized_bias > 2.0 else 0.002 if normalized_bias < -2.0 else 0

    # --- [新增指標 F] 量價背離偵測 (V-P Divergence) ---
    # 漲勢中若量能低於均量 20%，視為虛漲，增加向下阻力
    vp_divergence = -0.0025 if (change_pct > 0.5 and vol_ratio < 0.8) else 0

    # --- [新增指標 G] 波動率極度壓縮校正 (Vol Squeeze) ---
    atr_long_avg = df['ATR'].tail(60).mean()
    vol_gap_boost = 1.4 if (last['ATR'] < atr_long_avg * 0.75) else 1.0

    # --- [新增指標 H] 資金流向監控 (Simplified MFI) ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    # 判斷 14 日資金流入/流出
    flow_dir = np.where(tp > tp.shift(1), 1, -1)
    pos_mf = rmf.where(flow_dir > 0, 0).tail(14).sum()
    neg_mf = rmf.where(flow_dir < 0, 0).tail(14).sum()
    mfi_val = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-5))))
    # 資金背離邏輯
    mfi_drag = -0.0035 if (change_pct > 0.3 and mfi_val < 45) else 0

    # --- [新增指標 I] 乖離加速度 (Bias Velocity) ---
    bias_v = normalized_bias - (prev_c - prev['MA20'])/(prev['ATR']+1e-5)
    bias_accel = 0.0015 if (normalized_bias > 1.2 and bias_v > 0) else 0

    # --- [新增指標 J] 蔡金波動衰竭 (Chaikin Volatility Decay) ---
    # 計算 HL 差值的變動，偵測是否進入「高檔無力」或「低檔止跌」
    hl_ema = (df['High'] - df['Low']).ewm(span=10).mean()
    chv = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-5)
    # 高位波動率驟降通常是反轉訊號
    vol_exhaustion = -0.003 if (chv.iloc[-1] < -0.2 and change_pct > 0.5) else 0.002 if (chv.iloc[-1] < -0.2 and change_pct < -0.5) else 0

    # --- [新增指標 K] RSI 動能斜率 (RSI Momentum) ---
    rsi_s = df['RSI'].diff(3).iloc[-1]
    rsi_mom_boost = 0.0025 if (last['RSI'] > 50 and rsi_s > 5) else -0.0025 if (last['RSI'] < 50 and rsi_s < -5) else 0

    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    np.random.seed(42)
    sim_results = []
    
    # [核心連動公式最終注入] 加入主力力道矩陣 (whale_force / whale_dump)
    base_drift = (((int(precision) - 55) / 1000) * float(trend_weight) * ma_perfect_order + 
                  (rsi_div * 0.0025) + (chip_mom * 0.15) + (b_drift * 0.22) + 
                  exhaustion_drag + slope_decay + vol_bias_pull + vp_divergence + 
                  mfi_drag + bias_accel + vol_exhaustion + rsi_mom_boost + 
                  whale_force + whale_dump)
    
    for _ in range(1000):
        # 注入擠壓補償與波動壓縮擴張 (vol_gap_boost)
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
    
    # 診斷建議邏輯 (進階指標強化版)
    ma_check_list = [5, 10, 15, 20, 25, 30]
    above_ma_count = sum(1 for p in ma_check_list if curr_p > df['Close'].rolling(p).mean().iloc[-1])

    score = 0
    reasons = []
    
    # --- A. 趨勢與排列 ---
    if ma_perfect_order > 1.0: score += 2; reasons.append("多頭完美排列(飆股模式)")
    elif above_ma_count >= 5: score += 1.5; reasons.append(f"均線多頭排列")
    
    # --- B. 能量與背離 ---
    if is_squeezing: reasons.append("布林極度擠壓(即將噴發)")
    if exhaustion_drag < 0: score -= 0.5; reasons.append("漲勢背離力竭")
    
    # --- C. [新增] 慣性與引力監控 ---
    if slope_decay < 0: 
        score -= 0.3; reasons.append("均線慣性減速")
    if normalized_bias > 2.0: 
        score -= 0.5; reasons.append("波動超漲(引力修正)")
    elif normalized_bias < -2.0: 
        score += 0.5; reasons.append("波動超跌(引力支撐)")
    
    # --- [新增] 量價、資金與變盤監控 ---
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

    # --- D. 籌碼與共振 ---
    if whale_force > 0: score += 1.2; reasons.append("偵測大戶敲單進場")
    if whale_dump < 0: score -= 1.2; reasons.append("大戶棄守逃命跡象")
    if change_pct > 1.2 and vol_ratio > 1.3: score += 1; reasons.append("法人級放量攻擊")
    if b_drift > 0.003: score += 1; reasons.append("標本群體向上共振")
    
    status_map = {3: ("🚀 強力買入", "#FF3131"), 2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41"), -2: ("📉 偏空警戒", "#00FF41")}
    res = status_map.get(max(-2, min(3, int(score))), ("⚖️ 觀望中性", "#FFFF00"))
    
    adv = {k: {"buy": m * (1 - f_vol * v_comp * f * sens), "sell": m * (1 + f_vol * v_comp * f * sens)} for k, (m, f) in {"5日極短線建議": (df['Close'].rolling(5).mean().iloc[-1], 0.8), "10日短線建議": (df['Close'].rolling(10).mean().iloc[-1], 1.1), "20日波段建議": (last['MA20'], 1.5)}.items()}
    b_sum = {p: (curr_p - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5) for p in [5, 10, 20, 30]}
    
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), b_sum)
# --- 修改後 ---
def render_terminal(symbol, p_days, cp, tw_val, api_ttl, v_comp, ws_p, ws_w): # <-- 新增 ws_w 參數
    r_key = datetime.now().strftime("%Y-%m-%d %H:%M") 
    
    df, f_id = fetch_comprehensive_data(symbol, api_ttl * 60, r_key)
    if df is None: 
        st.error(f"❌ 讀取 {symbol} 失敗"); return

    # 1. 執行 AI 引擎
    final_p, final_tw, ai_v, ai_b, bias, f_vol, b_drift = auto_fine_tune_engine(df, cp, tw_val, v_comp)
    
    # 2. 執行預測運算
    pred_line, ai_recs, curr_p, open_p, prev_c, curr_v, change_pct, insight = perform_ai_engine(
        df, p_days, final_p, final_tw, ai_v, bias, f_vol, b_drift
    )
    
    # 新版呼叫：增加 ws_w，啟動 14:30 全清單自動補完與覆寫更新邏輯
    stock_accuracy = auto_sync_feedback(ws_p, ws_w, f_id, insight, cp, tw_val, v_comp, p_days, api_ttl)

    # 3. 視覺樣式定義
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

   # 4. 頂部標題與核心指標
    st.title(f"📊 {f_id} 台股AI預測系統") # 這裡改用 f_id (fetch函式回傳的代碼)

    # 5. 渲染 10 日橫向表格 (直接使用 stock_accuracy 變數，它是 Section 3 回傳的 DataFrame)
    # --- 修改後的寫法：限制寬度並緊縮間距 ---
    if stock_accuracy is not None and isinstance(stock_accuracy, pd.DataFrame):
        display_df = stock_accuracy.tail(10)
        
        # 限制整體表格寬度：只佔左邊 50%，右邊 50% 留空
        t_limit_col, _ = st.columns([0.5, 0.5])
        
        with t_limit_col:
            # 動態分配欄位權重：第一欄(標題)稍寬，其餘均分
            num_data = len(display_df)
            cols = st.columns([1.5] + [1] * num_data)
            
            with cols[0]:
                st.markdown("<p style='margin-bottom:8px; color:#888;'>日期</p>", unsafe_allow_html=True)
                st.markdown("<p style='color:#888;'>精準度</p>", unsafe_allow_html=True)
        
            for i, (_, row) in enumerate(display_df.iterrows()):
                with cols[i+1]:
                    st.markdown(f"<p style='margin-bottom:8px;'>{row['short_date']}</p>", unsafe_allow_html=True)
                    acc_val = row['accuracy_pct']
                    color = "#FF3131" if acc_val >= 95 else "#FFAC33"
                    st.markdown(f"<b style='color:{color}'>{acc_val:.1f}%</b>", unsafe_allow_html=True)
    else:
        st.info("💡 尚無歷史精準度數據，系統開始累積中...")

    # 6. AI 大腦說明接在表格下方
    st.caption(f"✨ AI 大腦：籌碼與動能分析 | 環境共振分析 | 技術面與乖離率評估 | 自我學習與反饋")

    c_p = "#FF3131" if change_pct >= 0 else "#00FF41"
    sign = "+" if change_pct >= 0 else ""
    m_cols = st.columns(5)
    metrics = [
        ("昨日收盤", f"{prev_c:.2f}", "#FFFFFF"), ("今日開盤", f"{open_p:.2f}", "#FFFFFF"), 
        ("當前價格", f"{curr_p:.2f}", c_p), ("今日漲跌", f"{sign}{change_pct:.2f}%", c_p), 
        ("成交 (張)", f"{int(curr_v/1000):,}", "#FFFF00")
    ]
    
    for i, (lab, val, col) in enumerate(metrics):
        with m_cols[i]: 
            st.markdown(f"<div class='info-box'><span style='color:#888; font-size:1.1rem; margin-bottom:5px;'>{lab}</span><b style='color:{col}; font-size:2.0rem; line-height:1;'>{val}</b></div>", unsafe_allow_html=True)

    # 5. 買賣點診斷區
    st.write(""); s_cols = st.columns(3)
    for i, (label, p) in enumerate(ai_recs.items()):
        with s_cols[i]: 
            st.markdown(f"<div class='diag-box'><b style='font-size:1.5rem; color:#FFFFFF;'>{label}</b><hr style='border:0.5px solid #444; width:80%; margin:10px 0;'><div style='font-size:1.2rem; color:#CCC;'>買入: <span style='color:#FF3131; font-weight:900; font-size:1.6rem;'>{p['buy']:.2f}</span></div><div style='font-size:1.2rem; color:#CCC;'>賣出: <span style='color:#00FF41; font-weight:900; font-size:1.6rem;'>{p['sell']:.2f}</span></div></div>", unsafe_allow_html=True)

    # 6. 補回所有線型標註 (含 MACD 與 KDJ)
    t_main = "■ 價格與均線 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#FF3131'>●</span> K線 <span style='color:#FFD700'><b>━━</b></span> 5MA <span style='color:#00F5FF'><b>━━</b></span> 10MA <span style='color:#FF00FF'><b>━━</b></span> 20MA <span style='color:#FF3131'><b>···</b></span> AI預測</span>"
    t_vol  = "■ 成交量分析 (張)"
    t_macd = "■ MACD 指標 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#FF3131'>■</span> 能量柱 <span style='color:#FFFFFF'><b>━━</b></span> DIF <span style='color:#FFA726'><b>━━</b></span> DEA</span>"
    t_kdj  = "■ KDJ 擺動指標 <span style='font-weight:normal; font-size:14px; color:#AAA;'>&nbsp;&nbsp; <span style='color:#00F5FF'><b>━━</b></span> K值 <span style='color:#FFFF00'><b>━━</b></span> D值 <span style='color:#E066FF'><b>━━</b></span> J值</span>"

    # 7. 繪製四層子圖
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.4, 0.15, 0.2, 0.25], 
        vertical_spacing=0.04, 
        subplot_titles=(t_main, t_vol, t_macd, t_kdj)
    )
    p_df = df.tail(90)
    
    # 7.1 主圖 (K線、均線、AI預測線)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], increasing_line_color='#FF3131', decreasing_line_color='#00FF41', showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], line=dict(color='#FFD700', width=2), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA10'], line=dict(color='#00F5FF', width=1.5), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MA20'], line=dict(color='#FF00FF', width=2), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=[p_df.index[-1] + timedelta(days=i) for i in range(1, p_days + 1)], y=pred_line, line=dict(color='#FF3131', width=3, dash='dash'), showlegend=False), 1, 1)
    
    # 7.2 量圖
    v_colors = ['#FF3131' if p_df['Close'].iloc[i] >= p_df['Open'].iloc[i] else '#00FF41' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume']/1000, marker_color=v_colors, showlegend=False), 2, 1)
    
    # 7.3 MACD 圖
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], marker_color=['#FF3131' if v >= 0 else '#00FF41' for v in p_df['Hist']], showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], line=dict(color='#FFFFFF', width=1.2), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], line=dict(color='#FFA726', width=1.2), showlegend=False), 3, 1)
    
    # 7.4 KDJ 圖
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['K'], line=dict(color='#00F5FF'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['D'], line=dict(color='#FFFF00'), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['J'], line=dict(color='#E066FF'), showlegend=False), 4, 1)

    # 8. 圖表外觀優化
    fig.update_layout(template="plotly_dark", height=880, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=50, b=50), paper_bgcolor='#000000', plot_bgcolor='#000000')
    
    # 確保子圖標題靠左對齊且為白色
    for i in fig['layout']['annotations']:
        i['x'] = 0; i['xanchor'] = 'left'; i['font'] = dict(size=14, color="#FFFFFF")

    st.plotly_chart(fig, use_container_width=True)

    # 9. AI 底部診斷建議 Box (微調版)
    
    # 計算近 10 日平均精準度文字
    if stock_accuracy is not None and isinstance(stock_accuracy, pd.DataFrame):
        avg_acc_text = f"🎯 10日平均精準度: {stock_accuracy['accuracy_pct'].mean():.1f}%"
    else:
        avg_acc_text = "🎯 實戰數據累積中"

    b_html = " | ".join([f"{k}D: <span style='color:{'#FF3131' if v >= 0 else '#00FF41'}'>{v:.2%}</span>" for k, v in insight[6].items()])
    
    st.markdown(f"""
        <div class='ai-advice-box'>
            <div class='confidence-tag'>{avg_acc_text}</div>
            <span style='font-size:1.5rem; color:{insight[2]}; font-weight:900;'>{insight[0]}</span>
            <hr style='border:0.5px solid #444; margin:10px 0;'>
            <p><b>AI診斷建議:</b> {insight[1]}</p>
            <p style='font-size:0.9rem; color:#8899A6;'>乖離率參考: {b_html}</p>
            <div style='background: #1C2128; padding: 12px; border-radius: 8px;'>
                <p style='color:#00F5FF; font-weight:bold; margin:0;'>🔮 AI 統一展望 (基準日: {df.index[-1].strftime('%Y/%m/%d')})：</p>
                <p style='font-size:1.8rem; color:#FFAC33; font-weight:900; margin:5px 0;'>預估隔日收盤價：{insight[3]:.2f}</p>
                <p style='color:#8899A6; margin:0;'>預估浮動區間：{insight[5]:.2f} ~ {insight[4]:.2f}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
def main():
    # --- 狀態保持邏輯 ---
    if 'user' not in st.session_state: 
        st.session_state.user = None  
    
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
        cp, api_ttl = int(s_map.get('global_precision', 55)), int(s_map.get('api_ttl_min', 1))
        tw_val = float(s_map.get('trend_weight', 1.0))
        v_comp = float(s_map.get('whale_sensitivity', 1.5))
    except Exception as e:
        st.error(f"🚨 資料庫連線失敗: {e}"); return

    if st.session_state.user is None:
        st.title("🚀 StockAI 台股預測系統")
        tab_login, tab_reg = st.tabs(["🔑 系統登入", "📝 註冊帳號"])
        with tab_login:
            u = st.text_input("請輸入帳號", key="login_u")
            p = st.text_input("請輸入密碼", type="password", key="login_p")
            if st.button("登入帳號", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf.empty and not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("❌ 驗證失敗")
        with tab_reg:
            new_u = st.text_input("新帳號", key="reg_u"); new_p = st.text_input("新密碼", type="password", key="reg_p")
            if st.button("提交註冊申請"):
                if new_u and new_p:
                    existing_users = pd.DataFrame(ws_u.get_all_records())
                    if not existing_users.empty and str(new_u) in existing_users['username'].astype(str).values:
                        st.error(f"❌ 帳號 '{new_u}' 已被註冊")
                    else:
                        ws_u.append_row([str(new_u), str(new_p)])
                        st.success("✅ 註冊成功")
    else:
        # --- 使用者儀表板 ---
        with st.expander("⚙️ :red[管理自選股清單(點擊開啟)]", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = all_w[all_w['username']==st.session_state.user]['stock_symbol'].tolist()
                target = st.selectbox("自選股清單", u_stocks if u_stocks else ["2330.TW"])
                ns = st.text_input("➕ 增加標的 (例: 2454.TW)")
                if st.button("加入清單"):
                    if ns:
                        new_stock = ns.upper().strip()
                        if len(u_stocks) < 20 and new_stock not in u_stocks:
                            ws_w.append_row([st.session_state.user, new_stock])
                            st.success(f"✅ {new_stock} 已加入"); st.rerun()
                
                if u_stocks:
                    if st.button(f"🗑️ 刪除目前標的 ({target})", use_container_width=True):
                        try:
                            # 關鍵修正：準確定位該用戶的該股票行
                            all_data = ws_w.get_all_values()
                            for i, row in enumerate(all_data):
                                if row[0] == st.session_state.user and row[1] == target:
                                    ws_w.delete_rows(i + 1); st.success("已移除"); st.rerun()
                        except: st.error("刪除失敗")

            # ✅ 修正後的 m2 區塊 (包含管理員戰情室)
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("---")
                    st.markdown("### 🛠️ 管理員戰情室")
                    r_key = datetime.now().strftime("%Y-%m-%d %H:%M")
                    temp_df, _ = fetch_comprehensive_data(target, api_ttl*60, r_key)
                    if temp_df is not None:
                        ai_p, ai_tw, ai_v, ai_b, _, _, _ = auto_fine_tune_engine(temp_df, cp, tw_val, v_comp)
                    else:
                        ai_p, ai_tw, ai_v, ai_b = cp, tw_val, 1.5, ["2330.TW", "2317.TW", "0050.TW"]
                    
                    b1 = st.text_input("1. 藍籌標本", ai_b[0] if len(ai_b)>0 else "2330.TW")
                    b2 = st.text_input("2. 成長標本", ai_b[1] if len(ai_b)>1 else "2317.TW")
                    b3 = st.text_input("3. 指數標本", ai_b[2] if len(ai_b)>2 else "0050.TW")
                    
                    cp = st.slider(f"靈敏度 (AI: {ai_p})", 0, 100, int(cp))
                    tw_val = st.number_input(f"權重 (AI: {ai_tw})", 0.5, 3.0, float(tw_val))
                    v_comp = st.slider(f"波動 (AI: {ai_v})", 0.5, 3.0, float(v_comp))
                    api_ttl = st.number_input("API TTL (分)", 1, 10, int(api_ttl))
                    
                    if st.button("💾 同步參數至雲端"):
                        ws_s.update_cell(2, 2, str(cp)); ws_s.update_cell(3, 2, str(api_ttl))
                        ws_s.update_cell(4, 2, b1); ws_s.update_cell(5, 2, b2); ws_s.update_cell(6, 2, b3)
                        ws_s.update_cell(7, 2, str(tw_val)); ws_s.update_cell(8, 2, str(v_comp))
                        st.success("✅ 更新成功"); st.rerun()
                
                st.write("")
                if st.button("🚪 登出系統", use_container_width=True): 
                    st.session_state.user = None; st.rerun()

        # ✅ 正確縮排：執行最終渲染與全清單對帳
        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p, ws_w)

if __name__ == "__main__":
    main()




