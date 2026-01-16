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
# 第一章：系統連線與環境配置 (Environment & Connection)
# =================================================================

def init_gspread():
    """
    段落：雲端權限初始化
    功能：支援 GCP 環境變數與 Streamlit Secrets 讀取 JSON 金鑰
    """
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        try:
            import streamlit as st
            creds_json = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
        except: pass
        
    if not creds_json:
        raise ValueError("環境變數 GCP_SERVICE_ACCOUNT_JSON 缺失")
    
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    return gspread.authorize(creds)


# =================================================================
# 第二章：市場數據獲取引擎 (Market Data Engine)
# =================================================================

def calculate_rsi(df, periods=14):
    """ 段落：技術指標庫 - RSI 相對強弱指標計算 """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def fetch_comprehensive_data(symbol):
    """ 
    段落：個股數據抓取與結構清洗
    功能：自動識別台股格式，並強制修復 yfinance 的 MultiIndex 欄位問題
    """
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s]
    if not (raw_s.endswith(".TW") or raw_s.endswith(".TWO")):
        search_list = [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) > 40:
                # --- [關鍵修復段落：處理 yfinance 新版多重索引] ---
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                # --------------------------------------------
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                return df, s
        except: continue
    return None, raw_s

def fetch_market_context():
    """ 段落：大盤環境抓取 - 獲取加權指數基準 """
    try:
        mkt = yf.download("^TWII", period="60d", interval="1d", auto_adjust=True, progress=False)
        # --- [關鍵修復段落：處理大盤多重索引] ---
        if isinstance(mkt.columns, pd.MultiIndex): 
            mkt.columns = mkt.columns.get_level_values(0)
        # ------------------------------------
        return mkt
    except: return None


# =================================================================
# 第三章：預測之神大腦 - 核心運算邏輯 (God Mode Intelligence)
# =================================================================

def god_mode_engine(df, symbol, mkt_df):
    """
    核心運算：處理所有數學模型並產出 A~AJ 欄位數據
    """
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    
    # --- [A] 大盤修正因子與個股 Beta 連動計算 ---
    mkt_trend, beta = 1.0, 1.0
    if mkt_df is not None:
        m_returns = mkt_df['Close'].pct_change().dropna()
        s_returns = df['Close'].pct_change().dropna()
        common = m_returns.index.intersection(s_returns.index)
        if len(common) > 10:
            beta = np.cov(s_returns[common], m_returns[common])[0,1] / (np.var(m_returns[common]) + 1e-9)
        mkt_ma20 = mkt_df['Close'].rolling(20).mean().iloc[-1]
        mkt_trend = 1.03 if mkt_df['Close'].iloc[-1] > mkt_ma20 else 0.97

    # --- [B] 乖離率體系預算 (準備 AD-AG 欄位) ---
    bias_list = []
    for n in [5, 10, 15, 20]:
        ma = df['Close'].rolling(n).mean().iloc[-1]
        b_val = round(((curr_p - ma) / (ma + 1e-9)) * 100, 2)
        bias_list.append(float(b_val))
    
    # --- [C] 戰略價格水位計算 (準備 G-X 欄位：18 個關鍵位) ---
    periods = [5, 10, 15, 20, 25, 30]
    buy_levels, sell_levels, resist_levels = [], [], []
    for p in periods:
        sub = df.tail(p)
        ma, std = sub['Close'].mean(), sub['Close'].std()
        b_p = (ma - (std * 1.5)) * 0.4 + sub['Low'].min() * 0.6
        s_p = ma + (std * 1.3)
        r_p = max(sub['High'].max(), ma + (std * 2.1))
        buy_levels.append(float(round(b_p, 2)))
        sell_levels.append(float(round(s_p, 2)))
        resist_levels.append(float(round(r_p, 2)))
    strategic_data = buy_levels + sell_levels + resist_levels

    # --- [D] 蒙地卡羅 7 天預測路徑 (準備 AA 欄位) ---
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

    # --- [E] 專家技術指標特徵 (準備 AH-AJ 欄位) ---
    atr = (df['High'].tail(14).max() - df['Low'].tail(14).min()) / 14
    vol_ratio = df['Volume'].iloc[-1] / (df['Volume'].tail(20).mean() + 1e-9)
    upside = pred_7d_list.max() - curr_p
    downside = curr_p - buy_levels[0]
    rr_ratio = round(float(upside / (abs(downside) + 1e-9)), 2)
    rsi_series = calculate_rsi(df)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    
    sentiment = "冷靜"
    if bias_list[0] > 7 or rsi_val > 75: sentiment = "過熱"
    elif bias_list[0] < -7 or rsi_val < 25: sentiment = "恐慌"
    expert_data = [float(round(atr, 2)), float(round(vol_ratio, 2)), float(rr_ratio), sentiment]

    # --- [F] AI 文字診斷與未來展望 (準備 AB, AC 欄位) ---
    chip_status = "資金流入" if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and vol_ratio > 1.2) else "籌碼穩定"
    insight = (f"【Oracle 診斷】{symbol} 目前趨勢偏{chip_status}。5日乖離 {bias_list[0]}%，"
               f"盈虧比評估為 {rr_ratio}。建議關注 5D 支撐位 {buy_levels[0]}。")
    outlook = f"AI 模擬未來 7 日目標價為 ${round(pred_7d_list[-1], 2)}，市場 Beta 係數為 {beta:.2f}。"

    return float(round(pred_7d_list[0], 2)), pred_path_str, insight, outlook, bias_list, strategic_data, expert_data


# =================================================================
# 第四章：自動化執行與數據同步 (Daily Sync Logic)
# =================================================================

def run_daily_sync():
    """
    段落：主程序循環
    功能：檢查時間、讀取 Watchlist、執行運算、封裝 36 欄位並上傳
    """
    try:
        # --- 1. 執行時間安全鎖 (台北時間 14:30) ---
        tw_tz = pytz.timezone('Asia/Taipei')
        now = datetime.now(tw_tz)
        if now.hour < 14 or (now.hour == 14 and now.minute < 30):
            print(f"⏳ 當前時間 {now.strftime('%H:%M')}，未達 14:30，跳過。")
            return

        client = init_gspread()
        sh = client.open("users")
        ws_p, ws_w = sh.worksheet("predictions"), sh.worksheet("watchlist")
        
        # --- 2. 獲取觀察名單 (注意：若超過 20 支應有警示) ---
        all_watchlists = ws_w.get_all_values()[1:]
        unique_symbols = set(str(row[1]).strip().upper() for row in all_watchlists if len(row) >= 2 and row[1])
        if not unique_symbols: return

        existing_rows = ws_p.get_all_values()
        mkt_df = fetch_market_context()

        # --- 3. 逐一標的處理循環 ---
        for symbol in unique_symbols:
            try:
                df, f_id = fetch_comprehensive_data(symbol)
                if df is None: continue
                data_date = df.index[-1].strftime("%Y-%m-%d")
                
                # 去重機制：同日同標的不重複寫入
                if any(len(row) >= 2 and row[0] == data_date and row[1] == f_id for row in existing_rows):
                    continue

                # 呼叫大腦運算
                p_next, path_str, insight, outlook, biases, s_data, expert_data = god_mode_engine(df, f_id, mkt_df)
                
                # --- 4. 數據封裝：精確對齊 A~AJ (共 36 欄) ---
                upload_row = [
                    data_date, f_id, p_next, round(p_next*0.985, 2), round(p_next*1.015, 2), "待更新"
                ] + s_data + [
                    0,          # Y: actual_close
                    0,          # Z: error_pct
                    path_str,   # AA: pred_path
                    insight,    # AB: ai_insight
                    outlook     # AC: ai_outlook
                ] + biases + expert_data
                
                # --- 5. 雲端寫入與速率緩衝 ---
                ws_p.append_row(upload_row)
                print(f"✅ {f_id} 分析同步完成 (A-AJ 欄位共 {len(upload_row)} 欄)")
                time.sleep(3) 

            except Exception as e:
                print(f"❌ {symbol} 處理失敗: {e}")

    except Exception as e:
        print(f"💥 系統異常: {e}")


# =================================================================
# 第五章：程式進入點 (Main Entry)
# =================================================================

if __name__ == "__main__":
    run_daily_sync()
