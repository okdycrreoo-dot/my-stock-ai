from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
import pytz
from datetime import datetime
from google.oauth2.service_account import Credentials

# =================================================================
# 第一章：初始化與環境連線 (第一章)
# =================================================================

def init_gspread():
    """ 
    初始化 Google Sheets 連線，確保在 Streamlit 與 Local 環境均可執行
    """
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if not creds_json:
        try:
            import streamlit as st
            creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
        except:
            pass
            
    if not creds_json:
        raise ValueError("CRITICAL ERROR: GCP_SERVICE_ACCOUNT_JSON 缺失，請檢查環境變數。")
    
    # 載入金鑰資訊
    info = json.loads(creds_json)
    
    # 設定存取範圍
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    # 建立憑證
    creds = Credentials.from_service_account_info(info, scopes=scope)
    
    # 授權連線
    client = gspread.authorize(creds)
    return client


# =================================================================
# 第二章：高階數據抓取引擎 (籌碼強化版)
# =================================================================

def calculate_rsi(df, periods=14):
    """ 
    計算 RSI 相對強弱指標，手動處理 1e-9 防止分母為零 
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_comprehensive_data(symbol):
    """ 
    抓取個股數據，並自動針對台股代號進行模糊搜尋 (.TW 或 .TWO) 
    """
    raw_s = str(symbol).strip().upper()
    
    if raw_s.endswith(".TW") or raw_s.endswith(".TWO"):
        search_list = [raw_s]
    else:
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]
        
    for s in search_list:
        try:
            print(f"📡 正在嘗試抓取 {s} 歷史數據...")
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            
            if df is not None and not df.empty and len(df) > 40:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                print(f"✅ 成功獲取 {s} 數據。")
                return df, s
        except Exception:
            continue
            
    print(f"❌ {raw_s} 在 .TW 與 .TWO 均無法獲取數據。")
    return None, raw_s


def fetch_market_context():
    """ 
    抓取大盤指數 (^TWII) 作為 AI 判斷的宏觀環境 
    """
    try:
        print(f"📡 正在獲取台股大盤數據...")
        mkt = yf.download("^TWII", period="60d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(mkt.columns, pd.MultiIndex): 
            mkt.columns = mkt.columns.get_level_values(0)
        return mkt
    except Exception as e:
        print(f"⚠️ 大盤數據獲取失敗: {e}")
        return None


def fetch_chip_data(symbol, token):
    """ 
    [新增] 從 FinMind 抓取三大法人近 3 日買賣超數據
    """
    import requests
    try:
        # 轉換格式：從 "2330.TW" 提取出 "2330"
        pure_id = symbol.split('.')[0]
        
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": pure_id,
            "token": token
        }
        
        print(f"📡 正在抓取 {pure_id} 三大法人籌碼面...")
        res = requests.get(url, params=parameter)
        data = res.json()
        
        if data.get('status') == 200 and data.get('data'):
            df_chip = pd.DataFrame(data['data'])
            # 取最近 3 個交易日
            recent_chip = df_chip.tail(3)
            # 計算淨買賣張數總和 (買進張數 - 賣出張數)
            net_total = recent_chip['buy'].sum() - recent_chip['sell'].sum()
            print(f"📊 {pure_id} 近三日法人淨買賣: {net_total} 張")
            return float(net_total)
            
        print(f"⚠️ {pure_id} 查無籌碼數據，回傳 0")
        return 0.0
    except Exception as e:
        print(f"❌ 籌碼抓取異常: {e}")
        return 0.0

# =================================================================
# 第三章：預測之神大腦 - AI 核心運算 (第三章)
# =================================================================

def god_mode_engine(df, symbol, mkt_df, chip_score=0.0):
    """
    AI 核心：執行 Beta 修正、多週期戰略水位、蒙地卡羅預測路徑與專家指標診斷。
    """
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    
    # --- [A] 大盤趨勢與 Beta 係數計算邏輯 ---
    mkt_trend = 1.0
    beta = 1.0
    
    if mkt_df is not None:
        # 計算個股與大盤收益率
        m_returns = mkt_df['Close'].pct_change().dropna()
        s_returns = df['Close'].pct_change().dropna()
        
        # 尋找共同交易日進行協方差運算
        common_idx = m_returns.index.intersection(s_returns.index)
        
        if len(common_idx) > 10:
            m_data = m_returns[common_idx]
            s_data = s_returns[common_idx]
            # 計算 Beta：Cov(s, m) / Var(m)
            covariance = np.cov(s_data, m_data)[0, 1]
            variance = np.var(m_data) + 1e-9
            beta = covariance / variance
        
        # 判斷大盤 20MA 趨勢 (趨勢加成)
        mkt_ma20 = mkt_df['Close'].rolling(20).mean().iloc[-1]
        if mkt_df['Close'].iloc[-1] > mkt_ma20:
            mkt_trend = 1.03
        else:
            mkt_trend = 0.97

    # --- [B] 乖離率計算 (AD, AE, AF, AG 欄位) ---
    bias_list = []
    for n in [5, 10, 15, 20]:
        ma_val = df['Close'].rolling(n).mean().iloc[-1]
        bias_val = ((curr_p - ma_val) / (ma_val + 1e-9)) * 100
        bias_list.append(float(round(bias_val, 2)))
    
    # --- [C] 戰略水位邏輯 (G 欄至 X 欄: 共 18 欄) ---
    # 循環 5, 10, 15, 20, 25, 30 日
    periods = [5, 10, 15, 20, 25, 30]
    buy_levels = []
    sell_levels = []
    resist_levels = []
    
    for p in periods:
        sub_df = df.tail(p)
        p_ma = sub_df['Close'].mean()
        p_std = sub_df['Close'].std()
        
        # 支撐位公式：結合標差與歷史低點
        support = (p_ma - (p_std * 1.5)) * 0.4 + sub_df['Low'].min() * 0.6
        # 壓力位公式：1.3 倍標差
        pressure = p_ma + (p_std * 1.3)
        # 強壓位公式：歷史高點與 2.1 倍標差取大值
        strong_res = max(sub_df['High'].max(), p_ma + (p_std * 2.1))
        
        buy_levels.append(float(round(support, 2)))
        sell_levels.append(float(round(pressure, 2)))
        resist_levels.append(float(round(strong_res, 2)))
        
    # 合併水位數據 (6+6+6 = 18 欄)
    strategic_data = buy_levels + sell_levels + resist_levels

    # --- [技術指標全掃描：計算 Tech Score] ---
    tech_score = 50  # 初始中性分
    try:
        # 1. 趨勢與動能組 (MACD, KDJ, RSI, DMI)
        macd = df.ta.macd()
        kdj = df.ta.kdj()
        rsi_val = df.ta.rsi(length=14).iloc[-1]
        adx = df.ta.adx()
        
        # 2. 能量與量價組 (OBV, NVI, PVI)
        obv_increasing = df.ta.obv().tail(5).is_monotonic_increasing
        nvi_val = df.ta.nvi().iloc[-1]
        nvi_prev = df.ta.nvi().iloc[-2]
        
        # 3. 複合指標 (BBI)
        bbi = (df['Close'].rolling(3).mean() + df['Close'].rolling(6).mean() + 
               df['Close'].rolling(12).mean() + df['Close'].rolling(24).mean()) / 4

        # --- 開始評分邏輯 ---
        if macd.iloc[-1, 0] > 0: tech_score += 8       # MACD DIF > 0
        if kdj.iloc[-1, 0] > kdj.iloc[-1, 1]: tech_score += 10  # K > D (金叉)
        if rsi_val > 50: tech_score += 5              # RSI 偏強
        if adx.iloc[-1, 1] > adx.iloc[-1, 2]: tech_score += 8  # +DI > -DI
        if obv_increasing: tech_score += 10            # 成交量能推升
        if nvi_val > nvi_prev: tech_score += 10        # 大戶能量 (NVI) 上升
        if curr_p > bbi.iloc[-1]: tech_score += 7      # 站上 BBI 多空線
        
    except:
        tech_score = 50 # 若計算失敗則維持中性
    
    # --- [D] 蒙地卡羅模擬 7 日路徑 (強化籌碼修正) ---
    np.random.seed(int(time.time()))
    volatility = df['Close'].pct_change().tail(20).std()
    
    # 籌碼動能加成：若法人大買，給予 1.02~1.15 的偏移加速
    # 我們設定 1000 張為一個基準門檻 (可根據股本調整)
    chip_boost = 1.0
    if chip_score > 500: # 買超超過 500 張
        chip_boost = 1.03 + min(chip_score / 10000, 0.12)
    elif chip_score < -500: # 賣超超過 500 張
        chip_boost = 0.97 - min(abs(chip_score) / 10000, 0.08)

    # 進化後的 drift 公式
    # 加入技術面修正因子 (Tech Boost)
    tech_boost = 1.0 + (tech_score - 50) / 1000 
    drift = (df['Close'].pct_change().tail(10).mean() * mkt_trend * chip_boost * tech_boost) - (bias_list[3] * 0.005)
    
    simulation_results = []
        
    # 執行 800 次路徑模擬
    for _ in range(800):
        temp_path = [curr_p]
        for _ in range(7):
            # 加入 Beta 敏感度修正
            random_shock = np.random.normal(drift, volatility * (1 + abs(beta-1)))
            
            # 計算下一日的原始預測價格
            next_p = temp_path[-1] * (1 + random_shock)
            
            # --- [台股專屬：10% 漲跌幅強制限縮] ---
            # 確保每一天的波動都不會超過前一天的 +-10%
            upper_limit = temp_path[-1] * 1.10
            lower_limit = temp_path[-1] * 0.90
            next_p = max(min(next_p, upper_limit), lower_limit)
            
            temp_path.append(next_p)
        simulation_results.append(temp_path[1:])
    
    # 取模擬平均路徑
    avg_path = np.mean(simulation_results, axis=0)
    # 轉為字串儲存
    path_string = ",".join([str(round(float(x), 2)) for x in avg_path])

    # --- [E] 專家級指標體系 (AH, AI, AJ, AK, AL 欄位) ---
    # ATR 波動指標
    atr_val = (df['High'].tail(14).max() - df['Low'].tail(14).min()) / 14
    # 量比指標
    volume_ratio = df['Volume'].iloc[-1] / (df['Volume'].tail(20).mean() + 1e-9)
    # 盈虧比評估
    max_upside = avg_path.max() - curr_p
    min_downside = curr_p - buy_levels[0]
    risk_reward = round(float(max_upside / (abs(min_downside) + 1e-9)), 2)
    
    # RSI 計算與情緒判定
    rsi_series = calculate_rsi(df)
    current_rsi = float(rsi_series.iloc[-1])
    
    market_sentiment = "冷靜"
    if bias_list[0] > 7 or current_rsi > 75:
        market_sentiment = "過熱"
    elif bias_list[0] < -7 or current_rsi < 25:
        market_sentiment = "恐慌"

    # --- 新增：AI 信心度計算 (對位 AL 欄) ---
    base_conf = 0.85
    # 根據風險回報比調整：R/R 高於 1.5 加分，低於 0.8 扣分
    conf_bonus = 0.05 if risk_reward > 1.5 else (-0.05 if risk_reward < 0.8 else 0)
    # 根據 RSI 穩定度調整：過於極端則信心下降
    conf_adj = -0.1 if current_rsi > 85 or current_rsi < 15 else 0.02
    final_confidence = round(min(max(base_conf + conf_bonus + conf_adj, 0.5), 0.98), 2)
        
    # 封裝專家數據 (5 欄位：ATR, 量比, 盈虧比, 情緒, 信心度)
    expert_metrics = [
        float(round(atr_val, 2)), 
        float(round(volume_ratio, 2)), 
        float(risk_reward), 
        market_sentiment,
        final_confidence
    ]

    # --- [F] AI 綜合診斷文本 (這裡就是你要加的地方) ---
    mkt_view = "看多" if mkt_trend > 1 else "保守"
    
    # 新增：籌碼狀態判定邏輯
    if chip_score > 1500:
        chip_msg = "🔥 法人強勢進場"
    elif chip_score > 500:
        chip_msg = "✅ 法人小幅買超"
    elif chip_score < -1500:
        chip_msg = "💀 法人集體拋售"
    elif chip_score < -500:
        chip_msg = "⚠️ 法人小幅賣超"
    else:
        chip_msg = "⚖️ 籌碼中性穩定"

    # 判斷當日是否接近漲跌停 (增強診斷文本)
    price_change_ratio = abs((curr_p - last['Open']) / (last['Open'] + 1e-9))
    limit_msg = " [!觸及極端限制]" if price_change_ratio > 0.098 else ""

    # 封裝診斷文本
    # 範例：在診斷開頭加上 [分:xx]
    diag_insight = (f"【Oracle 評分:{tech_score}】{symbol}({chip_msg}){limit_msg}。大盤環境{mkt_view}(Beta:{beta:.2f})。 "
                    f"5日乖離 {bias_list[0]}%，盈虧比 {risk_reward}。")
   
    forecast_outlook = f"AI 模擬 7 日目標價為 ${round(avg_path[-1], 2)}，短期支撐位參考 {buy_levels[0]}。"

    # 最後統一回傳所有結果
    return float(round(avg_path[0], 2)), path_string, diag_insight, forecast_outlook, bias_list, strategic_data, expert_metrics


# =================================================================
# 第四章：自動同步作業 (加入保護期停機邏輯)
# =================================================================

def run_daily_sync(target_symbol=None):
    try:
        FINMIN_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMS0yNyAxNTo0NDo0MSIsInVzZXJfaWQiOiJrZCIsImVtYWlsIjoib2tkeWNycmVvb0BnbWFpbC5jb20iLCJpcCI6IjEzNi4yMjYuMjQxLjk2In0.JUMtA2-Y98F-AUMgRtIa11o56WmX1Yx6T40q5RgM4oE" # 貼上你的 Token
        # --- [核心保護機制：23:00 - 14:30 大腦強制熔斷] ---
        # 取得台北時間
        tz = pytz.timezone('Asia/Taipei')
        now_time = datetime.now(tz)
        current_time = now_time.time()
        
        # 設定保護時間界限
        start_lock = datetime.strptime("23:50", "%H:%M").time()
        end_lock = datetime.strptime("14:00", "%H:%M").time()
        
        # 判斷是否處於保護期
        if current_time >= start_lock or current_time <= end_lock:
            print(f"🚫 【大腦絕對保護中】")
            print(f"目前台北時間：{now_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("保護期規則：每日 23:50 至隔日 14:00 期間，大腦拒絕任何分析、計算與寫入動作。")
            return # 強制結束，不執行下方所有代碼
        # -----------------------------------------------

        # 只有在非保護期，大腦才會繼續往下執行
        today_str = now_time.strftime('%Y-%m-%d')
        is_urgent = bool(target_symbol)

        # 開始連線 (這之後才會動到 Google Sheets)
        client = init_gspread()
        spreadsheet = client.open("users")
        ws_predict = spreadsheet.worksheet("predictions")
        ws_watch = spreadsheet.worksheet("watchlist")
        
        # 1. 抓取名單 (支援 3105.TWO 等上櫃代碼)
        symbols_set = set()
        if is_urgent:
            symbols_set.add(target_symbol.strip().upper())
        else:
            watch_data = ws_watch.get_all_values()[1:]
            for row in watch_data:
                if len(row) >= 2 and row[1]:
                    symbols_set.add(str(row[1]).strip().upper())
        
        if not symbols_set:
            print("❌ 名單為空。")
            return

        # 2. 回填校準 (1-16 的 F, Y, Z)
        all_logs = ws_predict.get_all_values()
        for i, row in enumerate(all_logs[1:], 1):
            if len(row) >= 6 and "待更新" in row[5] and row[0] != today_str:
                try:
                    h_df, _ = fetch_comprehensive_data(row[1])
                    if h_df is not None and len(h_df) >= 3:
                        actual_now = round(float(h_df['Close'].iloc[-1]), 2)
                        y_val_fixed = round(float(h_df['Close'].iloc[-3]), 2)
                        err = round(((actual_now - float(row[2])) / float(row[2])) * 100, 2)
                        ws_predict.update_cell(i+1, 6, actual_now)
                        ws_predict.update_cell(i+1, 25, y_val_fixed)
                        ws_predict.update_cell(i+1, 26, err)
                        time.sleep(1)
                except: continue

        # 3. 執行今日新預測 (1-19 補齊 Y 欄)
        market_df = fetch_market_context()
        if len(symbols_set) > 20:
            print(f"⚠️ 提醒：Watchlist 已達 {len(symbols_set)} 支，超過上限！")

        for sym in symbols_set:
            try:
                stock_df, final_id = fetch_comprehensive_data(sym)
                if stock_df is None: continue
                # --- [2. 在這裡插入：抓取籌碼分數] ---
                # 呼叫第二章新增的函數
                chip_score = fetch_chip_data(final_id, FINMIN_TOKEN)
                # --- [3. 修改：將 chip_score 傳入大腦] ---
                # 原本是 god_mode_engine(stock_df, final_id, market_df)
                # 現在多加一個 chip_score 參數
                current_logs = ws_predict.get_all_values()
                exists_idx = next((idx+1 for idx, r in enumerate(current_logs) if r[0] == today_str and r[1] == final_id), None)

                p_val, p_path, p_diag, p_out, p_bias, p_levels, p_experts = god_mode_engine(stock_df, final_id, market_df, chip_score)
                y_val = round(float(stock_df['Close'].iloc[-2]), 2) if len(stock_df) >= 2 else round(float(stock_df['Close'].iloc[-1]), 2)

                if not exists_idx:
                    # 原有的新增邏輯 (保持不變)
                    row_data = [today_str, final_id, p_val, round(p_val*0.985, 2), round(p_val*1.015, 2), "待更新"] + \
                               (p_levels + [0]*18)[:18] + [y_val, 0, p_path, p_diag, p_out] + \
                               (p_bias + [0]*4)[:4] + (p_experts + [0]*5)[:5]
                    ws_predict.append_row(row_data)
                    print(f"✅ {final_id} 新增成功，AI 信心度: {p_experts[4]}")
                else:
                    # --- 優化：即使存在，也檢查並補填數據 ---
                    # 1. 補填 Y 欄 (第 25 欄)
                    ws_predict.update_cell(exists_idx, 25, y_val)
                    
                    # 2. 檢查 AL 欄 (第 38 欄) 是否為空或 0
                    existing_row = current_logs[exists_idx-1]
                    # 判斷 AL 欄 (索引 37) 是否沒有數據
                    if len(existing_row) <= 37 or not str(existing_row[37]).strip() or str(existing_row[37]) == "0":
                        conf_val = p_experts[4]
                        ws_predict.update_cell(exists_idx, 38, conf_val) # 第 38 欄就是 AL
                        print(f"⚡ {final_id} 已存在，但補填 AL 欄信心度: {conf_val}")
                    else:
                        print(f"⚡ {final_id} 已存在且已有數據，僅更新 Y 欄。")
                
                time.sleep(2)
            except Exception as e:
                print(f"❌ {sym} 處理異常: {e}")

    except Exception as e:
        print(f"💥 全域錯誤: {e}")
# =================================================================
# 第五章：啟動入口 (EntryPoint)
# =================================================================

if __name__ == "__main__":
    target_stock = os.environ.get("TARGET_SYMBOL", "").strip().upper()
    if target_stock:
        print(f"🚀 即時分析啟動: {target_stock}")
        run_daily_sync(target_stock)
    else:
        print("📅 定時掃描任務啟動。")
        run_daily_sync()
