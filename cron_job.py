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

# -----------------------------------------------------------------
# 1. 初始化與連線
# -----------------------------------------------------------------
# --- 修改 init_gspread 函數 ---
def init_gspread():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        # 在 Streamlit 環境中，嘗試從 st.secrets 抓取
        import streamlit as st
        creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
        
    if not creds_json:
        raise ValueError("環境變數 GCP_SERVICE_ACCOUNT_JSON 缺失")
    
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    # ⚠️ 這裡改用 from_service_account_info
    creds = Credentials.from_service_account_info(info, scopes=scope)
    return gspread.authorize(creds)
# -----------------------------------------------------------------
# 2. 數據引擎：增加 RSI 計算
# -----------------------------------------------------------------
def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def fetch_comprehensive_data(symbol):
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s]
    if not (raw_s.endswith(".TW") or raw_s.endswith(".TWO")):
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) > 40:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                return df, s
        except: continue
    return None, raw_s

def fetch_market_context():
    try:
        mkt = yf.download("^TWII", period="60d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
        return mkt
    except: return None

# -----------------------------------------------------------------
# 3. 預測之神核心：專家級權策大腦 (數據格式優化版)
# -----------------------------------------------------------------
def god_mode_engine(df, symbol, mkt_df):
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    
    # [A] 大盤修正因子
    mkt_trend, beta = 1.0, 1.0
    if mkt_df is not None:
        m_returns = mkt_df['Close'].pct_change().dropna()
        s_returns = df['Close'].pct_change().dropna()
        common = m_returns.index.intersection(s_returns.index)
        if len(common) > 10:
            beta = np.cov(s_returns[common], m_returns[common])[0,1] / (np.var(m_returns[common]) + 1e-9)
        mkt_ma20 = mkt_df['Close'].rolling(20).mean().iloc[-1]
        mkt_trend = 1.03 if mkt_df['Close'].iloc[-1] > mkt_ma20 else 0.97

    # [B] 指標計算與乖離率
    bias_list = []
    for n in [5, 10, 15, 20]:
        ma = df['Close'].rolling(n).mean().iloc[-1]
        b_val = round(((curr_p - ma) / (ma + 1e-9)) * 100, 2)
        bias_list.append(float(b_val)) # 強制轉為 float 確保 Sheets 識別
    
    # [C] 戰略水位 (30D) - 修正後的價格計算邏輯
    periods = [5, 10, 15, 20, 25, 30]
    buy_levels, sell_levels, resist_levels = [], [], []
    for p in periods:
        sub = df.tail(p)
        ma, std = sub['Close'].mean(), sub['Close'].std()
        
        # 支撐/壓力位計算 (確保輸出是純數字價格)
        b_p = (ma - (std * 1.5)) * 0.4 + sub['Low'].min() * 0.6
        s_p = ma + (std * 1.3)
        r_p = max(sub['High'].max(), ma + (std * 2.1))
        
        buy_levels.append(float(round(b_p, 2)))
        sell_levels.append(float(round(s_p, 2)))
        resist_levels.append(float(round(r_p, 2)))
    
    strategic_data = buy_levels + sell_levels + resist_levels

    # [D] 7天預測軌跡
    np.random.seed(int(time.time()))
    f_vol = df['Close'].pct_change().tail(20).std()
    drift = (df['Close'].pct_change().tail(10).mean() * mkt_trend) - (bias_list[3] * 0.005)
    
    sim_paths = []
    for _ in range(800):
        path = [curr_p]
        for _ in range(7):
            change = np.random.normal(drift, f_vol * (1 + abs(beta-1)))
            path.append(path[-1] * (1 + change))
        sim_paths.append(path[1:])
    
    pred_7d_list = np.mean(sim_paths, axis=0)
    pred_path_str = ",".join([str(round(float(x), 2)) for x in pred_7d_list])

    # [E] 專家級指標維度 (AF-AI)
    atr = (df['High'].tail(14).max() - df['Low'].tail(14).min()) / 14
    vol_ratio = df['Volume'].iloc[-1] / (df['Volume'].tail(20).mean() + 1e-9)
    
    # 盈虧比穩定化計算
    upside = pred_7d_list.max() - curr_p
    downside = curr_p - buy_levels[0]
    rr_ratio = round(float(upside / (abs(downside) + 1e-9)), 2)
    
    rsi_series = calculate_rsi(df)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    
    sentiment = "冷靜"
    if bias_list[0] > 7 or rsi_val > 75: sentiment = "過熱"
    elif bias_list[0] < -7 or rsi_val < 25: sentiment = "恐慌"
    
    expert_data = [float(round(atr, 2)), float(round(vol_ratio, 2)), float(rr_ratio), sentiment]

    # [F] AI 綜合診斷報告 (AA 欄)
    chip_status = "資金流入" if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and vol_ratio > 1.2) else "籌碼穩定"
    mkt_text = "看多" if mkt_trend > 1 else "保守"
    best_inds = "MACD, Bias, Bollinger" if abs(bias_list[0]) > 3 else "MA, RSI, KDJ"
    
    insight = (f"【Oracle 診斷】{symbol} 目前趨勢偏{chip_status}。大盤環境{mkt_text}(Beta:{beta:.2f})。 "
               f"AI 依股性選擇最佳指標：{best_inds}。 5日乖離 {bias_list[0]}%，"
               f"盈虧比評估為 {rr_ratio}。建議關注 5D 支撐位 {buy_levels[0]}。")

    return float(round(pred_7d_list[0], 2)), pred_path_str, insight, bias_list, strategic_data, expert_data

# -----------------------------------------------------------------
# 4. 全局自動化同步邏輯 (交易日自動對齊修正版)
# -----------------------------------------------------------------
def run_daily_sync():
    try:
        tw_tz = pytz.timezone('Asia/Taipei')
        now = datetime.now(tw_tz)
        
        # [時間鎖已註解] 方便凌晨或週末手動測試，正式上線後可解除註解
          if now.hour < 14 or (now.hour == 14 and now.minute < 30):
              print(f"⏳ 當前時間 {now.strftime('%H:%M')}，未達 14:30，跳過。")
              return

        client = init_gspread()
        sh = client.open("users")
        ws_p = sh.worksheet("predictions")
        ws_w = sh.worksheet("watchlist")
        
        # 1. 抓取所有使用者的 Watchlist 並去重
        all_watchlists = ws_w.get_all_values()[1:]
        unique_symbols = set(str(row[1]).strip().upper() for row in all_watchlists if len(row) >= 2 and row[1])
        
        if not unique_symbols:
            print("❌ Watchlist 為空，無須分析。")
            return

        # 獲取 predictions 表目前所有數據，用於檢查是否重複
        existing_rows = ws_p.get_all_values()

        print(f"🚀 啟動 Oracle 引擎：預計分析 {len(unique_symbols)} 支股票...")
        mkt_df = fetch_market_context()

        for symbol in unique_symbols:
            try:
                df, f_id = fetch_comprehensive_data(symbol)
                if df is None:
                    print(f"❓ 無法獲取 {symbol} 數據，跳過。")
                    continue
                
                # [核心修正] 使用 K 線數據最後一天的日期作為存檔基準日 (例如 2026-01-16)
                data_date = df.index[-1].strftime("%Y-%m-%d")
                
                # 2. 精準去重：檢查該股票在「該交易日」是否已經有分析結果
                is_done = False
                for row in existing_rows:
                    if len(row) >= 2 and row[0] == data_date and row[1] == f_id:
                        is_done = True
                        break
                
                if is_done:
                    print(f"⏩ {f_id} 於 {data_date} 的分析已存在，跳過。")
                    continue
                
                # 執行預測之神大腦 (回傳包含 expert_data)
                p_next, path_str, insight, biases, s_data, e_data = god_mode_engine(df, f_id, mkt_df)
                
                # 3. 準備上傳數據列 (A-AI 欄，總計 35 欄)
                # 使用 data_date 確保日期標籤與數據來源一致
                upload_row = [
                    data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待收盤更新"
                ] + s_data + [0] + [path_str, insight] + biases + e_data
                
                ws_p.append_row(upload_row)
                print(f"🔮 {f_id} 分析完成 (基準日: {data_date})。")
                
                # 同步更新本地比對清單，避免同一批次內意外重複
                existing_rows.append(upload_row)
                
                # 速率限制，保護 Google Sheets API
                time.sleep(3) 

            except Exception as e:
                print(f"❌ 分析 {symbol} 失敗: {e}")

    except Exception as e:
        print(f"💥 核心邏輯發生異常: {e}")

if __name__ == "__main__":
    run_daily_sync()
