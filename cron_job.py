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
# 第一章：初始化與連線 (Environment & Connection)
# =================================================================

def init_gspread():
    """ 
    初始化 Google Sheets 連線，支援 Streamlit Secrets 與環境變數 
    """
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if not creds_json:
        # 在 Streamlit 環境中，嘗試從 st.secrets 抓取
        try:
            import streamlit as st
            creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
        except:
            pass
        
    if not creds_json:
        raise ValueError("無法找到 GCP_SERVICE_ACCOUNT_JSON，請檢查 Secrets 設置。")
    
    info = json.loads(creds_json)
    # 設定權限範圍
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = Credentials.from_service_account_info(info, scopes=scope)
    return gspread.authorize(creds)


# =================================================================
# 第二章：市場數據獲取引擎 (Market Data Engine)
# =================================================================

def calculate_rsi(df, periods=14):
    """ 
    計算 RSI 指標，包含 1e-9 的極小值修正，避免除以零 
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fetch_comprehensive_data(symbol):
    """ 
    抓取個股數據，支援台股 (.TW / .TWO) 自動補全 
    """
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s]
    
    # 自動判定台股後置碼
    if not (raw_s.endswith(".TW") or raw_s.endswith(".TWO")):
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) > 40:
                # 處理 yfinance 可能回傳的 MultiIndex 欄位
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                # 選取核心欄位並轉為 float
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                return df, s
        except Exception as e:
            continue
            
    return None, raw_s


def fetch_market_context():
    """ 
    抓取大盤指數 (^TWII) 作為趨勢基準 
    """
    try:
        mkt = yf.download("^TWII", period="60d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(mkt.columns, pd.MultiIndex): 
            mkt.columns = mkt.columns.get_level_values(0)
        return mkt
    except:
        return None


# =================================================================
# 第三章：預測之神大腦 - 核心運算 (God Mode Intelligence)
# =================================================================

def god_mode_engine(df, symbol, mkt_df):
    """ 
    核心運算引擎：產出包含戰略水位、乖離率、蒙地卡羅與專家指標的完整數據 
    """
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    
    # --- [A] 大盤修正因子與 Beta 計算 ---
    mkt_trend, beta = 1.0, 1.0
    if mkt_df is not None:
        m_returns = mkt_df['Close'].pct_change().dropna()
        s_returns = df['Close'].pct_change().dropna()
        # 取交集日期
        common = m_returns.index.intersection(s_returns.index)
        if len(common) > 10:
            # 計算 Beta 係數
            beta = np.cov(s_returns[common], m_returns[common])[0,1] / (np.var(m_returns[common]) + 1e-9)
        
        # 判斷大盤 20MA 趨勢
        mkt_ma20 = mkt_df['Close'].rolling(20).mean().iloc[-1]
        mkt_trend = 1.03 if mkt_df['Close'].iloc[-1] > mkt_ma20 else 0.97

    # --- [B] 乖離率體系 (AD, AE, AF, AG 欄位) ---
    bias_list = []
    for n in [5, 10, 15, 20]:
        ma = df['Close'].rolling(n).mean().iloc[-1]
        b_val = round(((curr_p - ma) / (ma + 1e-9)) * 100, 2)
        bias_list.append(float(b_val))
    
    # --- [C] 戰略水位 (G 欄至 X 欄：共 18 個價格點) ---
    # 包含 5, 10, 15, 20, 25, 30 日的 支撐、壓力、強壓
    periods = [5, 10, 15, 20, 25, 30]
    buy_levels, sell_levels, resist_levels = [], [], []
    
    for p in periods:
        sub = df.tail(p)
        ma, std = sub['Close'].mean(), sub['Close'].std()
        
        # 支撐點：結合標準差與區間低點 (0.4/0.6 權重)
        b_p = (ma - (std * 1.5)) * 0.4 + sub['Low'].min() * 0.6
        # 壓力點：1.3 倍標準差
        s_p = ma + (std * 1.3)
        # 強力壓力點：區間最高與 2.1 倍標差取最大值
        r_p = max(sub['High'].max(), ma + (std * 2.1))
        
        buy_levels.append(float(round(b_p, 2)))
        sell_levels.append(float(round(s_p, 2)))
        resist_levels.append(float(round(r_p, 2)))
    
    # 按照 G-L (Buy), M-R (Sell), S-X (Resist) 排序
    strategic_data = buy_levels + sell_levels + resist_levels

    # --- [D] 7天蒙地卡羅路徑預測 (AA 欄位) ---
    np.random.seed(int(time.time()))
    f_vol = df['Close'].pct_change().tail(20).std()
    # 飄移率考慮大盤趨勢與 20 日乖離修正
    drift = (df['Close'].pct_change().tail(10).mean() * mkt_trend) - (bias_list[3] * 0.005)
    
    sim_paths = []
    for _ in range(800):
        path = [curr_p]
        for _ in range(7):
            # 隨機擾動考慮 Beta 放大效應
            change = np.random.normal(drift, f_vol * (1 + abs(beta-1)))
            path.append(path[-1] * (1 + change))
        sim_paths.append(path[1:])
    
    pred_7d_list = np.mean(sim_paths, axis=0)
    # 轉為逗號分隔字串
    pred_path_str = ",".join([str(round(float(x), 2)) for x in pred_7d_list])

    # --- [E] 專家級指標 (AH, AI, AJ 欄位) ---
    # ATR (波動率)
    atr = (df['High'].tail(14).max() - df['Low'].tail(14).min()) / 14
    # 量比
    vol_ratio = df['Volume'].iloc[-1] / (df['Volume'].tail(20).mean() + 1e-9)
    # 盈虧比計算
    upside = pred_7d_list.max() - curr_p
    downside = curr_p - buy_levels[0]
    rr_ratio = round(float(upside / (abs(downside) + 1e-9)), 2)
    
    # RSI 計算
    rsi_series = calculate_rsi(df)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    
    # 市場情緒判定
    sentiment = "冷靜"
    if bias_list[0] > 7 or rsi_val > 75: 
        sentiment = "過熱"
    elif bias_list[0] < -7 or rsi_val < 25: 
        sentiment = "恐慌"
        
    expert_data = [
        float(round(atr, 2)), 
        float(round(vol_ratio, 2)), 
        float(rr_ratio), 
        sentiment
    ]

    # --- [F] AI 診斷與展望 (AB, AC 欄位) ---
    chip_status = "資金流入" if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and vol_ratio > 1.2) else "籌碼穩定"
    mkt_text = "看多" if mkt_trend > 1 else "保守"
    
    insight = (f"【Oracle 診斷】{symbol} 目前趨勢偏{chip_status}。大盤環境{mkt_text}(Beta:{beta:.2f})。 "
               f"5日乖離 {bias_list[0]}%，盈虧比評估為 {rr_ratio}。")
    
    outlook = f"AI 模擬未來 7 日目標價為 ${round(pred_7d_list[-1], 2)}，建議關注 5D 支撐位 {buy_levels[0]}。"

    # 回傳：預測價, 路徑, 診斷, 展望, 乖離清單, 水位清單, 專家數據
    return float(round(pred_7d_list[0], 2)), pred_path_str, insight, outlook, bias_list, strategic_data, expert_data


# =================================================================
# 第四章：全自動同步邏輯 (Daily Sync Logic)
# =================================================================

def run_daily_sync():
    """ 
    執行每日同步：讀取 Watchlist -> 運算 -> 寫入 Predictions (精確對齊 36 欄) 
    """
    try:
        tw_tz = pytz.timezone('Asia/Taipei')
        now = datetime.now(tw_tz)
        
        # 判定交易日更新時間 (14:30 後)
        if now.hour < 14 or (now.hour == 14 and now.minute < 30):
            print(f"⏳ 當前時間 {now.strftime('%H:%M')}，尚未達 14:30 更新門檻。")
            return

        client = init_gspread()
        sh = client.open("users")
        ws_p = sh.worksheet("predictions")
        ws_w = sh.worksheet("watchlist")
        
        # 1. 抓取觀察名單
        all_watchlists = ws_w.get_all_values()[1:]
        unique_symbols = set(str(row[1]).strip().upper() for row in all_watchlists if len(row) >= 2 and row[1])
        
        # --- 數量上限提醒邏輯 (您的指定要求：20支) ---
        if len(unique_symbols) > 20:
            print(f"⚠️ 警告：目前名單共有 {len(unique_symbols)} 支標的，已超過您設定的 20 支上限。")
        
        if not unique_symbols:
            print("❌ Watchlist 為空。")
            return

        # 2. 獲取現有數據用於重複檢查
        existing_rows = ws_p.get_all_values()
        mkt_df = fetch_market_context()

        print(f"🚀 開始分析 {len(unique_symbols)} 支標的...")

        for symbol in unique_symbols:
            try:
                df, f_id = fetch_comprehensive_data(symbol)
                if df is None:
                    continue
                
                # 使用數據最後日期
                data_date = df.index[-1].strftime("%Y-%m-%d")
                
                # 精確去重
                is_done = any(len(row) >= 2 and row[0] == data_date and row[1] == f_id for row in existing_rows)
                if is_done:
                    print(f"⏩ {f_id} 在 {data_date} 已分析，跳過。")
                    continue

                # 執行運算
                p_next, p_path, insight, outlook, biases, s_data, e_data = god_mode_engine(df, f_id, mkt_df)
                
                # --- [數據封裝：A-AJ 共 36 欄位] ---
                # A-F: 基礎資訊
                row_基础 = [data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待更新"]
                # G-X: 18欄水位 (s_data)
                # Y, Z: 實際收盤與誤差 (預留佔位)
                row_佔位 = [0, 0]
                # AA-AC: AI 預測軌跡與文本
                row_AI文本 = [p_path, insight, outlook]
                # AD-AJ: 乖離與指標 (biases + e_data)
                row_指標 = biases + e_data
                
                # 組合最終橫列
                upload_row = row_基础 + s_data + row_佔位 + row_AI文本 + row_指標
                
                # 物理長度檢查
                if len(upload_row) == 36:
                    ws_p.append_row(upload_row)
                    print(f"✅ {f_id} 分析完成 (基準日: {data_date})。")
                else:
                    print(f"❌ {f_id} 欄位異常 (當前長度: {len(upload_row)}，預期 36)")
                
                # API 限流保護
                time.sleep(3) 

            except Exception as e:
                print(f"❌ 分析 {symbol} 失敗: {e}")

    except Exception as e:
        print(f"💥 核心邏輯發生異常: {e}")


# =================================================================
# 第五章：程式進入點 (Main Entry Point)
# =================================================================

if __name__ == "__main__":
    # 執行主程序
    run_daily_sync()
