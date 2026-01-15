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

# =================================================================
# 第二章：數據引擎 (Data Engine)
# =================================================================

def fetch_comprehensive_data(stock_id, period_seconds=3600):
    """
    [2-1 & 2-2 整合段落]
    """
    try:
        # --- [2-1 段] 自動識別與格式化代碼 ---
        f_id = str(stock_id).upper().strip()
        
        if not (f_id.endswith(".TW") or f_id.endswith(".TWO")):
            # 優先嘗試上市格式
            test_id = f_id + ".TW"
            ticker = yf.Ticker(test_id)
            df = ticker.history(period="1mo")
            
            if df.empty:
                # 若上市查無資料，嘗試上櫃格式
                test_id = f_id + ".TWO"
                ticker = yf.Ticker(test_id)
                df = ticker.history(period="1mo")
                
            f_id = test_id
        else:
            # 若已帶後綴，直接抓取
            ticker = yf.Ticker(f_id)
            df = ticker.history(period="1mo")

        # 檢查最終是否有數據，若無則回傳空值
        if df.empty:
            return None, None

        # --- [2-2 段] 顯性籌碼因子幕後計算 (隱藏於後台) ---
        # 目的：透過成交量與價格變動的連動性，量化法人/大戶的推動力道
        df['Price_Change'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        
        # 指標公式：當價格變動與量能變動同步放大，代表籌碼力道強化
        df['Inst_Force'] = df['Price_Change'] * df['Vol_Change'] * 100
        
        # 填補計算首列產生的空值，確保數據完整性
        df = df.fillna(0)

        # 回傳包含籌碼因子的數據集與正確代碼
        return df, f_id

    except Exception as e:
        return None, None

# =================================================================
# 第三章：自動對帳與反饋系統 (Feedback System)
# =================================================================

# --- [3-1 段] auto_sync_feedback 函數與時間判定邏輯 ---
def auto_sync_feedback(ws_p, f_id, insight):
    try:
        recs = ws_p.get_all_records()
        df_p = pd.DataFrame(recs)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # 14:30 收盤判定邏輯 (14*60 + 30 = 870 分鐘)
        is_after_market = (now.hour * 60 + now.minute) >= 870
        is_weekend = now.weekday() >= 5

        # --- [3-2 段] 歷史對帳邏輯：回填目標日已過的實際股價 ---
        for i, row in df_p.iterrows():
            # 若 actual_close 欄位為空，且該列記錄的預測目標日期已到達或已過(<=今天)
            if str(row.get('actual_close', '')).strip() == "" and str(row.get('date', '')) <= today_str:
                target_date = row['date']
                # 抓取該目標日的收盤數據 (end_date 設為隔日以確保抓到當天)
                end_date = (pd.to_datetime(target_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                h = yf.download(row['symbol'], start=target_date, end=end_date, progress=False)
                
                if not h.empty:
                    # 處理 yfinance 可能產生的 MultiIndex 欄位
                    act_df = h.copy()
                    if isinstance(act_df.columns, pd.MultiIndex):
                        act_df.columns = act_df.columns.get_level_values(0)
                    
                    act_close = float(act_df['Close'].iloc[-1])
                    pred_close = float(row['pred_close'])
                    
                    # 更新試算表：第 6 欄為實際收盤價，第 7 欄為誤差率
                    ws_p.update_cell(i + 2, 6, round(act_close, 2))
                    err_val = (act_close - pred_close) / pred_close
                    ws_p.update_cell(i + 2, 7, f"{err_val:.2%}")

        # --- [3-3 段] 新預測數據回填與命中率計算 ---
        # 14:30 收盤後且非週末才寫入新預測
        if is_after_market and not is_weekend:
            next_bus_day = now + timedelta(days=1)
            while next_bus_day.weekday() >= 5:
                next_bus_day += timedelta(days=1)
            next_day_str = next_bus_day.strftime("%Y-%m-%d")

            if not any((str(r.get('date')) == next_day_str and r.get('symbol') == f_id) for r in recs):
                new_row = [next_day_str, f_id, round(insight[3], 2), round(insight[5], 2), round(insight[4], 2), "", ""]
                ws_p.append_row(new_row)
        
        # 取得最近 10 筆已對帳數據並計算精確準確率
        df_stock = df_p[(df_p['symbol'] == f_id) & (df_p['actual_close'] != "")].copy()
        accuracy_history = []
        hit_text = "🎯 數據累積中"
        
        if not df_stock.empty:
            df_recent = df_stock.tail(10)
            for _, row in df_recent.iterrows():
                try:
                    act = float(row['actual_close'])
                    pred = float(row['pred_close'])
                    # 計算準確率：1 - |(實際-預測)/預測|
                    acc_val = (1 - abs(act - pred) / pred) * 100
                    acc_val = max(0, min(100, acc_val)) # 限制在 0-100%
                    
                    accuracy_history.append({
                        "date": str(row['date'])[-5:], 
                        "acc_val": f"{acc_val:.1f}%",
                        "color": "#FF3131" if acc_val >= 98 else "#FFFFFF" # 98% 以上顯示紅色
                    })
                except:
                    continue
            
            # 計算區間命中率文字
            hit = sum((df_recent['actual_close'].astype(float) >= df_recent['range_low'].astype(float)) & 
                      (df_recent['actual_close'].astype(float) <= df_recent['range_high'].astype(float)))
            hit_text = f"🎯 此股近期區間命中率: {(hit/len(df_recent))*100:.1f}%"
        
        return hit_text, accuracy_history
    except Exception as e:
        return f"🎯 同步中...", []
# =================================================================
# 第四章：AI 微調引擎 (Fine-tune Engine)
# =================================================================
def auto_fine_tune_engine(df, cp, tw_val, v_comp, env_panic=1.0):
    """
    負責吸收顯性籌碼、計算波動權重與生成推薦參數。
    """
    try:
        # --- [安全性檢查] 確保進入邏輯前指標已存在 ---
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        
        latest = df.iloc[-1]
        price_now = float(latest['Close'])
        
        # --- [4-1] 顯性籌碼力道提取 ---
        # 提取法人力道指標，若無則預設為 0
        inst_force = latest.get('Inst_Force', 0)
        v_curr = latest['Volume']
        v_avg5 = df['Volume'].tail(5).mean()
        vol_ratio = v_curr / (v_avg5 + 1e-5)
        
        # --- [4-2] 多維度波動與趨勢權重 ---
        rets = df['Close'].pct_change().dropna()
        v_p = [5, 10, 15, 20, 25, 30]
        v_w = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        v_vals = [rets.tail(p).std() for p in v_p]
        # 計算加權波動率並結合環境恐慌因子
        f_vol = sum(v * w for v, w in zip(v_vals, v_w)) * env_panic
        
        tw_adj = 0.8 if env_panic > 1.0 else 1.0
        # 核心公式：將籌碼力道融入趨勢權重 (final_tw)
        final_tw = max(0.5, min(2.5, 1.0 + (rets.tail(5).mean() * 15 + inst_force * 0.5) * min(1.5, vol_ratio) * tw_adj))
        
        # --- [4-3] 乖離偏好與漂移參數生成 ---
        b_periods = [5, 10, 15, 20, 25, 30]
        b_weights = [0.35, 0.2, 0.15, 0.1, 0.1, 0.1]
        # 計算多週期加權乖離率
        bias_list = [((price_now - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5)) for p in b_periods]
        bias_val = sum(b * w for b, w in zip(bias_list, b_weights))
        
        # 根據波動率決定模型精度 (Precision)
        final_p = (45 if f_vol > 0.02 else 75 if f_vol < 0.008 else 60)
        if env_panic > 1.0: final_p = int(final_p * 0.85)

        # 波動補償 ai_v 計算
        high_low_range = (df['High'] - df['Low']).tail(5).mean() / price_now
        ai_v = 1.3 if (high_low_range > 0.035 or abs(inst_force) > 0.8) else 2.1 if high_low_range < 0.015 else 1.7
        
        b_drift = 0.0 # 預設標桿漂移
        
        # 回傳 7 個參數以對接主程式 (final_p, final_tw, ai_v, ai_b, bias, f_vol, b_drift)
        return int(final_p), round(final_tw, 2), ai_v, bias_val, bias_val, f_vol, b_drift

    except Exception as e:
        # 保底數據，確保程式不因任何意外中斷
        return 50, 1.0, 1.7, 0.0, 0.0, 0.01, 0.0
        
# =================================================================
# 第五章：AI 預測運算核心 (AI Core Engine)
# =================================================================
def perform_ai_engine(df, p_days, precision, trend_weight, v_comp, bias, f_vol, b_drift):
    """
    [5-1 ~ 5-6 段] 完整的蒙地卡羅路徑演算法與多空評分系統
    """
    # --- [安全性修復] 確保 MA 系列欄位存在，解決 KeyError ---
    if 'MA20' not in df.columns: df['MA20'] = df['Close'].rolling(20).mean()
    if 'MA5' not in df.columns: df['MA5'] = df['Close'].rolling(5).mean()
    if 'MA10' not in df.columns: df['MA10'] = df['Close'].rolling(10).mean()
    if 'ATR' not in df.columns: df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    sens = (int(precision) / 55)
    curr_p = float(last['Close'])
    prev_c = float(prev['Close'])
    curr_v = int(last['Volume'])
    change_pct = ((curr_p - prev_c) / prev_c) * 100

    # --- [5-1 段] 主力力道矩陣 ---
    v_avg20 = df['Volume'].tail(20).mean() 
    vol_ratio = curr_v / (v_avg20 + 0.1)
    whale_force = (change_pct * 0.002) if (change_pct > 2.0 and vol_ratio > 1.5) else 0
    whale_dump = (change_pct * 0.0015) if (change_pct < -2.0 and vol_ratio > 1.5) else 0
    chip_mom = (change_pct / 100) * vol_ratio * 1.5 if (change_pct > 0.5 and vol_ratio > 1.2) else (change_pct / 100)

    # --- [5-2 段] 進階指標 A-C (布林擠壓、多空排列) ---
    std_20 = df['Close'].rolling(20).std()
    bb_width = (std_20 * 4) / (df['MA20'] + 1e-5) 
    is_squeezing = bb_width.iloc[-1] < bb_width.tail(20).mean() * 0.92
    squeeze_boost = 1.35 if is_squeezing else 1.0

    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    ma_perfect_order = 1.25 if (last['MA5'] > last['MA10'] > last['MA20'] > ma60) else 1.0

    # --- [5-5 段] 蒙地卡羅模擬運算邏輯 ---
    np.random.seed(42)
    sim_results = []
    
    # 核心漂移率計算 (包含所有微調參數)
    base_drift = (((int(precision) - 55) / 1000) * float(trend_weight) * ma_perfect_order + 
                  (chip_mom * 0.15) + (b_drift * 0.22) + whale_force + whale_dump)
    
    vol_contract = last['ATR'] / (df['ATR'].tail(10).mean() + 0.001)
    
    for _ in range(1000):
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
    
    # --- [5-6 段] 診斷建議與多空評分系統 ---
    score = 0
    reasons = []
    if ma_perfect_order > 1.0: score += 2; reasons.append("多頭完美排列")
    if is_squeezing: reasons.append("布林極度擠壓")
    if whale_force > 0: score += 1.2; reasons.append("偵測大戶敲單進場")
    if not reasons: reasons.append("走勢處於整理區間")

    # 映射最終狀態
    status_map = { 2: ("🚀 強力買入", "#FF3131"), 1: ("📈 偏多操作", "#FF7A7A"), 
                   0: ("⚖️ 觀望中性", "#FFFF00"), -1: ("📉 偏空警戒", "#00FF41") }
    res = status_map.get(max(-1, min(2, int(score))), ("⚖️ 觀望中性", "#FFFF00"))
    
    # 準備回傳數據
    adv = { "5日建議": {"buy": curr_p * 0.985, "sell": curr_p * 1.015} }
    b_sum = {p: (curr_p - df['Close'].rolling(p).mean().iloc[-1]) / (df['Close'].rolling(p).mean().iloc[-1] + 1e-5) for p in [5, 10, 20, 30]}
    
    return pred_prices, adv, curr_p, float(last['Open']), prev_c, curr_v, change_pct, (res[0], " | ".join(reasons), res[1], next_close, next_close + (std_val * 1.5), next_close - (std_val * 1.5), b_sum)b_sum)
    
# =================================================================
# 第六章：終端渲染引擎 (Render Terminal)
# =================================================================

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
    st.title(f"📊 {f_id} 台股AI預測系統")
    
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

    # --- [6-5 段] 底部 AI 診斷建議盒與展望預測輸出 ---
    b_html = " | ".join([f"{k}D: <span style='color:{'#FF3131' if v >= 0 else '#00FF41'}'>{v:.2%}</span>" for k, v in insight[6].items()])
    st.markdown(f"""
        <div class='ai-advice-box'>
            <div class='confidence-tag'>{stock_accuracy}</div>
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

# =================================================================
# 第七章：主程式邏輯與權限控管 (Main Logic)
# =================================================================

# --- [7-1 段] main() 函數初始化與逾時邏輯 ---
def main():
    if 'user' not in st.session_state: st.session_state.user, st.session_state.last_active = None, time.time()
    if st.session_state.user and (time.time() - st.session_state.last_active > 3600): st.session_state.user = None
    st.session_state.last_active = time.time()
    
    # --- [7-2 段] get_gsheets_connection 函數與授權 ---
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
        tw_val, v_comp = float(s_map.get('trend_weight', 1.0)), float(s_map.get('vol_comp', 1.5))
    except Exception as e:
        st.error(f"🚨 資料庫連線失敗: {e}"); return

    if st.session_state.user is None:
        st.title("🚀 StockAI 台股預測系統")
        tab_login, tab_reg = st.tabs(["🔑 系統登入", "📝 註冊帳號"])
        # --- [7-3 段] 登入分頁邏輯 (tab_login) ---
        with tab_login:
            u = st.text_input("請輸入帳號", key="login_u")
            p = st.text_input("請輸入密碼", type="password", key="login_p")
            if st.button("登入帳號", use_container_width=True):
                udf = pd.DataFrame(ws_u.get_all_records())
                if not udf.empty and not udf[(udf['username'].astype(str)==u) & (udf['password'].astype(str)==p)].empty:
                    st.session_state.user = u; st.rerun()
                else: st.error("❌ 驗證失敗")
        # --- [7-4 段] 註冊分頁邏輯 (tab_reg) ---
        with tab_reg:
            new_u = st.text_input("新帳號", key="reg_u")
            new_p = st.text_input("新密碼", type="password", key="reg_p")
            if st.button("提交註冊申請"):
                if not new_u or not new_p:
                    st.error("❌ 帳號或密碼不能為空白")
                else:
                    udf = pd.DataFrame(ws_u.get_all_records())
                    if not udf.empty and str(new_u) in udf['username'].astype(str).values:
                        st.error(f"⚠️ 帳號 '{new_u}' 已被註冊，請嘗試其他名稱")
                    else:
                        ws_u.append_row([str(new_u), str(new_p)])
                        st.success("✅ 註冊成功！現在可以切換至登入分頁。")
    
    else:
        # --- [7-5 段] 使用者自選股管理 (新增/刪除) ---
        with st.expander("⚙️ :red[管理自選股清單(點擊開啟)]", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                # 1. 讀取目前的自選清單
                all_w = pd.DataFrame(ws_w.get_all_records())
                u_stocks = []
                if not all_w.empty:
                    u_stocks = all_w[all_w['username'] == st.session_state.user]['stock_symbol'].tolist()
                
                target = st.selectbox("自選股清單", u_stocks if u_stocks else ["尚未新增"])
                
                # 2. 新增股票邏輯 (加入 20 支上限檢查)
                ns = st.text_input("➕ 輸入股票代號 (例: 2454)")
                if st.button("加入到自選股清單"):
                    if ns:
                        # 自動判定上市/上櫃並補上後綴 (.TW / .TWO)
                        _, final_s_code = fetch_comprehensive_data(ns, 3600)
                        
                        if final_s_code:
                            # --- 上限與重複檢查邏輯 ---
                            if len(u_stocks) >= 20:
                                st.error(f"🚫 自選股已達上限 (20 支)，請先刪除舊標的再新增。")
                            elif final_s_code in u_stocks:
                                st.warning(f"⚠️ {final_s_code} 已經在您的清單中囉！")
                            else:
                                ws_w.append_row([st.session_state.user, final_s_code])
                                st.success(f"✅ 已新增 {final_s_code}")
                                st.rerun()
                        else:
                            st.error("❌ 找不到該標的，請確認代號是否正確")
                    else:
                        st.info("💡 請先輸入代號")
                
                # 3. 刪除股票邏輯
                if u_stocks:
                    st.write("")
                    if st.button(f"🗑️ 刪除目前標的 ({target})", use_container_width=True):
                        try:
                            # 精確刪除：必須帳號與代號同時符合
                            all_w_full = pd.DataFrame(ws_w.get_all_records())
                            row_to_del = all_w_full[(all_w_full['username'] == st.session_state.user) & 
                                                    (all_w_full['stock_symbol'] == target)].index
                            
                            if not row_to_del.empty:
                                # gspread row index starts at 1, DataFrame at 0, plus 1 for header
                                ws_w.delete_rows(int(row_to_del[0]) + 2)
                                st.success(f"✅ {target} 已從您的清單移除")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ 刪除失敗: {e}")
            # --- [7-6 段] 管理員專屬戰情室 (參數調整與同步) ---
            with m2:
                p_days = st.number_input("預測天數", 1, 30, 7)
                if st.session_state.user == "okdycrreoo":
                    st.markdown("---")
                    st.markdown("### 🛠️ 管理員戰情室")
                    temp_df, _ = fetch_comprehensive_data(target, api_ttl*60)
                    ai_res = auto_fine_tune_engine(temp_df, cp, tw_val, v_comp) if temp_df is not None else (cp, tw_val, v_comp, ("2330", "2382", "00878"), 0, 0, 0)
                    ai_p, ai_tw, ai_v, ai_b = ai_res[0], ai_res[1], ai_res[2], ai_res[3]
                    
                    b1 = st.text_input(f"1. 基準藍籌股", ai_b[0])
                    b2 = st.text_input(f"2. 高波動成長股", ai_b[1])
                    b3 = st.text_input(f"3. 指數 ETF 標本", ai_b[2])
                    
                    st.write("")
                    new_p = st.slider(f"系統靈敏度", 0, 100, ai_p)
                    new_tw = st.number_input(f"趨勢權重參數", 0.5, 3.0, ai_tw)
                    new_v = st.slider(f"波動補償係數", 0.5, 3.0, ai_v)
                    new_ttl = st.number_input(f"Google API 連線時間", 1, 10, api_ttl)
                    
                    if st.button("💾 同步 AI 推薦參數至雲端"):
                        ws_s.update_cell(2, 2, str(new_p)); ws_s.update_cell(3, 2, str(new_ttl))
                        ws_s.update_cell(4, 2, b1); ws_s.update_cell(5, 2, b2); ws_s.update_cell(6, 2, b3)
                        ws_s.update_cell(7, 2, str(new_tw)); ws_s.update_cell(8, 2, str(new_v))
                        st.success("✅ 雲端配置已更新"); st.rerun()
                
                st.write("")
                if st.button("🚪 登出 StockAI 系統", use_container_width=True): st.session_state.user = None; st.rerun()

        render_terminal(target, p_days, cp, tw_val, api_ttl, v_comp, ws_p)

if __name__ == "__main__":
    main()


















