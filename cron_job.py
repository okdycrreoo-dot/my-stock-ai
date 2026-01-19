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
# 第二章：高階數據抓取引擎 (第二章)
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
    抓取個股數據，並自動針對台股代號 (.TW / .TWO) 進行模糊搜尋 
    """
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s]
    
    # 自動補全台股後置碼邏輯
    if not (raw_s.endswith(".TW") or raw_s.endswith(".TWO")):
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]
        
    for s in search_list:
        try:
            print(f"📡 正在抓取 {s} 歷史數據...")
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            
            if df is not None and not df.empty and len(df) > 40:
                # 處理 yfinance 可能產生的 MultiIndex 欄位結構
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                # 強制轉換資料型態為 float 確保計算不報錯
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                return df, s
        except Exception as e:
            print(f"⚠️ {s} 抓取嘗試失敗: {e}")
            continue
            
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


# =================================================================
# 第三章：預測之神大腦 - AI 核心運算 (第三章)
# =================================================================

def god_mode_engine(df, symbol, mkt_df):
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

    # --- [D] 蒙地卡羅模擬 7 日路徑 (AA 欄位) ---
    np.random.seed(int(time.time()))
    # 波動率使用最近 20 日標準差
    volatility = df['Close'].pct_change().tail(20).std()
    # 飄移率計算：近期 10 日均值 * 大盤係數 - 乖離率修正
    drift = (df['Close'].pct_change().tail(10).mean() * mkt_trend) - (bias_list[3] * 0.005)
    
    simulation_results = []
    # 執行 800 次路徑模擬
    for _ in range(800):
        temp_path = [curr_p]
        for _ in range(7):
            # 加入 Beta 敏感度修正
            random_shock = np.random.normal(drift, volatility * (1 + abs(beta-1)))
            temp_path.append(temp_path[-1] * (1 + random_shock))
        simulation_results.append(temp_path[1:])
    
    # 取模擬平均路徑
    avg_path = np.mean(simulation_results, axis=0)
    # 轉為字串儲存
    path_string = ",".join([str(round(float(x), 2)) for x in avg_path])

    # --- [E] 專家級指標體系 (AH, AI, AJ, AK 欄位) ---
    # ATR 波動指標
    atr_val = (df['High'].tail(14).max() - df['Low'].tail(14).min()) / 14
    # 量比指標 (當日成交量 / 20日平均量)
    volume_ratio = df['Volume'].iloc[-1] / (df['Volume'].tail(20).mean() + 1e-9)
    # 盈虧比評估 (預期漲幅 / 預期回撤)
    max_upside = avg_path.max() - curr_p
    min_downside = curr_p - buy_levels[0]
    risk_reward = round(float(max_upside / (abs(min_downside) + 1e-9)), 2)
    
    # RSI 計算與情緒判定
    rsi_series = calculate_rsi(df)
    current_rsi = float(rsi_series.iloc[-1])
    
    # AI 情緒邏輯 (這會放在 AK 欄位)
    market_sentiment = "冷靜"
    if bias_list[0] > 7 or current_rsi > 75:
        market_sentiment = "過熱"
    elif bias_list[0] < -7 or current_rsi < 25:
        market_sentiment = "恐慌"
        
    # 封裝專家數據 (4 欄位)
    expert_metrics = [
        float(round(atr_val, 2)), 
        float(round(volume_ratio, 2)), 
        float(risk_reward), 
        market_sentiment
    ]

    # --- [F] AI 綜合診斷文本 (AB, AC 欄位) ---
    money_flow = "資金流入" if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and volume_ratio > 1.2) else "籌碼穩定"
    mkt_view = "看多" if mkt_trend > 1 else "保守"
    
    diag_insight = (f"【Oracle 診斷】{symbol} 目前趨勢偏{money_flow}。大盤環境{mkt_view}(Beta:{beta:.2f})。 "
                    f"5日乖離 {bias_list[0]}%，盈虧比 {risk_reward}。")
    
    forecast_outlook = f"AI 模擬 7 日目標價為 ${round(avg_path[-1], 2)}，短期支撐位參考 {buy_levels[0]}。"

    # 回傳結果集
    return float(round(avg_path[0], 2)), path_string, diag_insight, forecast_outlook, bias_list, strategic_data, expert_metrics


# =================================================================
# 第四章：自動同步作業 (精確 A-AK 37 欄位 - 含舊資料回填邏輯)
# =================================================================

def run_daily_sync(target_symbol=None):
    try:
        tz = pytz.timezone('Asia/Taipei')
        now_time = datetime.now(tz)
        today_str = now_time.strftime('%Y-%m-%d')
        
        is_urgent = (target_symbol is not None and target_symbol != "")

        # 時間檢查
        if not is_urgent:
            if now_time.hour < 14 or (now_time.hour == 14 and now_time.minute < 30):
                print(f"⌛ 定時任務：目前時間 {now_time.strftime('%H:%M')}，未達更新時間，不執行。")
                return

        client = init_gspread()
        spreadsheet = client.open("users")
        ws_predict = spreadsheet.worksheet("predictions")
        ws_watch = spreadsheet.worksheet("watchlist")
        
        # 1. 抓取名單
        symbols_set = set()
        if is_urgent:
            symbols_set.add(str(target_symbol).strip().upper())
        else:
            watch_data = ws_watch.get_all_values()[1:]
            for row in watch_data:
                if len(row) >= 2 and row[1]:
                    symbols_set.add(str(row[1]).strip().upper())
        
        if not symbols_set:
            print("❌ 名單為空，終止同步。")
            return

        # 2. 【核心功能：回填校準 - 同步回補 Y 欄版】
        print("🔍 正在執行回填校準：補齊 F(Status), Y(Actual), Z(Error)...")
        all_logs = ws_predict.get_all_values()
        
        COL_F_STATUS = 6   
        COL_Y_ACTUAL = 25  
        COL_Z_ERROR = 26   

        for i, row in enumerate(all_logs):
            if i == 0: continue 
            current_status = str(row[COL_F_STATUS-1]).strip()
            
            if "待更新" in current_status:
                old_date = row[0]
                old_sym = row[1]
                if old_date == today_str: continue

                try:
                    old_pred_price = float(row[2])
                    ticker_ob = yf.Ticker(old_sym)
                    
                    # 抓取開獎價 (今日成交價)
                    actual_now = round(float(ticker_ob.fast_info['last_price']), 2)
                    # 抓取參考價 (今日看到的昨日收盤價，補回 Y 欄)
                    prev_close_to_fix = round(float(ticker_ob.fast_info['previous_close']), 2)

                    if actual_now > 0:
                        error_val = round(((actual_now - old_pred_price) / old_pred_price) * 100, 2)
                        row_num = i + 1
                        
                        # 同步寫入 F, Y, Z 三個欄位
                        ws_predict.update_cell(row_num, COL_F_STATUS, actual_now) 
                        time.sleep(1.2)
                        ws_predict.update_cell(row_num, COL_Y_ACTUAL, prev_close_to_fix) # 補填 Y 欄
                        time.sleep(1.2)
                        ws_predict.update_cell(row_num, COL_Z_ERROR, error_val)     
                        
                        print(f"✅ {old_sym} 已同步補齊 Y 欄與 F 欄數據")
                        time.sleep(2.5) 
                except Exception as e:
                    print(f"⚠️ 校準錯誤: {e}")
                    
        # 3. 執行今日新預測 (具備自動補漏洞與偵測功能)
        market_df = fetch_market_context()
        for sym in symbols_set:
            try:
                # 每一輪循環都獲取最新 logs，確保偵測最精準
                current_logs = ws_predict.get_all_values()
                stock_df, final_id = fetch_comprehensive_data(sym)
                if stock_df is None: continue

                # --- 【自動確認與補漏邏輯】 ---
                existing_row_idx = -1
                is_data_perfect = False
                
                for idx, row_data in enumerate(current_logs):
                    # 匹配日期與代號
                    if len(row_data) >= 2 and row_data[0] == today_str and row_data[1] == final_id:
                        existing_row_idx = idx + 1 
                        # 檢查第 37 欄 (AK欄) 是否有值，確保資料不是殘缺的
                        if len(row_data) >= 37 and str(row_data[36]).strip() != "":
                            is_data_perfect = True
                        break

                # 只有資料完整時才跳過
                if is_data_perfect and not is_urgent:
                    print(f"⏩ {final_id} 今日數據已完整填寫，跳過。")
                    continue
                
                # 如果有殘缺數據（例如上次執行到一半斷掉），先自動刪除舊列
                if existing_row_idx != -1:
                    print(f"🛠️ {final_id} 偵測到殘缺數據，正在自動清除並重新修復...")
                    ws_predict.delete_row(existing_row_idx)
                    time.sleep(2) 

                # --- [執行 AI 預測核心] ---
                p_val, p_path, p_diag, p_out, p_bias, p_levels, p_experts = god_mode_engine(stock_df, final_id, market_df)
                
                # 【關鍵補強】立刻抓取 Y 欄需要的「昨日收盤價」
                ticker_now = yf.Ticker(final_id)
                # 這就是你要求的：1/19 執行時抓到 1/16 的收盤價
                yesterday_close_val = round(float(ticker_now.fast_info['previous_close']), 2)
                
                # A-F: 基本資訊 (F 欄填待更新)
                col_base = [today_str, final_id, p_val, round(p_val*0.985, 2), round(p_val*1.015, 2), "待更新"]
                
                # G-X: 戰略水位 (18 欄位)
                col_levels = (list(p_levels) + [0]*18)[:18] 
                
                # Y-Z: 【修正重點】Y 欄填入剛剛抓到的昨日收盤價，Z 欄初始為 0
                col_calib = [yesterday_close_val, 0] 
                
                # AA-AK: 其餘 AI 文本與專家指標
                col_ai_txt = [p_path, p_diag, p_out]
                col_bias = (list(p_bias) + [0]*4)[:4]
                col_expert = (list(p_experts) + [0]*4)[:4]

                # 最終拼裝
                final_upload_row = col_base + col_levels + col_calib + col_ai_txt + col_bias + col_expert
                
                if len(final_upload_row) == 37:
                    ws_predict.append_row(final_upload_row)
                    print(f"✅ {final_id} 今日預測自動同步/修復成功 (37欄)。")
                else:
                    print(f"❌ {final_id} 拼裝異常，欄位數為: {len(final_upload_row)}")
                
                time.sleep(3) # 保護 API

            except Exception as e:
                print(f"❌ 標的 {sym} 處理異常: {e}")

    except Exception as e:
        print(f"💥 程式執行核心錯誤: {e}")


# =================================================================
# 第五章：啟動入口 (EntryPoint)
# =================================================================

if __name__ == "__main__":
    # 1. 取得目標代號 (由 GitHub Actions 傳入)
    target_stock = os.environ.get("TARGET_SYMBOL", "").strip().upper()

    # 2. 執行同步邏輯
    if target_stock:
        print(f"🚀 偵測到即時分析請求，目標標的: {target_stock}")
        run_daily_sync(target_stock)
    else:
        # 如果是定時任務 (沒傳 TARGET_SYMBOL)，執行全清單掃描
        print("📅 偵測到定時任務啟動，將執行全清單掃描。")
        run_daily_sync()
