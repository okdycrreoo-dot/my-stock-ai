import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
import pytz
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# =================================================================
# 1. 核心初始化：Google Sheets 連線
# =================================================================
def init_gspread():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("環境變數 GCP_SERVICE_ACCOUNT_JSON 缺失")
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    return gspread.authorize(creds)

# =================================================================
# 2. 數據引擎：全週期指標抓取 (5D - 30D)
# =================================================================
def fetch_comprehensive_data(symbol):
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s] if (raw_s.endswith(".TW") or raw_s.endswith(".TWO")) else [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        try:
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) > 35:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                
                # 計算基礎均線
                for p in [5, 10, 15, 20, 25, 30]:
                    df[f'MA{p}'] = df['Close'].rolling(p).mean()
                
                return df.dropna(), s
        except: continue
    return None, raw_s

# =================================================================
# 3. AI 大腦：30天戰略地圖與壓力位計算
# =================================================================
def perform_strategic_ai_engine(df, p_days=30):
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    periods = [5, 10, 15, 20, 25, 30]
    
    strategic_data = [] # 存放 18 個數據：Buy(6), Sell(6), Resistance(6)
    
    # --- A. 計算 6 個週期的關鍵價位 ---
    buy_levels = []
    sell_levels = []
    resist_levels = []
    
    for p in periods:
        sub = df.tail(p)
        ma = sub['Close'].mean()
        std = sub['Close'].std()
        h_max = sub['High'].max()
        l_min = sub['Low'].min()
        
        # 建議買價：均線扣除標準差與近期低點加權
        buy_p = (ma - (std * 1.5)) * 0.4 + l_min * 0.6
        # 建議賣價：均線加上標準差
        sell_p = ma + (std * 1.2)
        # 壓力價格：最高價與極端波動軌道的最大值
        resist_p = max(h_max, ma + (std * 2.1))
        
        buy_levels.append(round(buy_p, 2))
        sell_levels.append(round(sell_p, 2))
        resist_levels.append(round(resist_p, 2))
    
    strategic_data = buy_levels + sell_levels + resist_levels

    # --- B. 蒙地卡羅預測 (保留原本 7 天預測邏輯作為精確參考) ---
    rets = df['Close'].pct_change().tail(20)
    f_vol = rets.std()
    
    sim_results = []
    for _ in range(1000):
        noise = np.random.normal(0.0002, f_vol * 1.7, 7)
        path = [curr_p]
        for n in noise:
            path.append(path[-1] * (1 + n))
        sim_results.append(path[1:])
    
    pred_7d = np.mean(sim_results, axis=0)[0] # 明日預測值
    std_7d = np.std([p[0] for p in sim_results])
    
    return round(pred_7d, 2), round(pred_7d - std_7d*1.5, 2), round(pred_7d + std_7d*1.5, 2), strategic_data

# =================================================================
# 4. 主流程：自動化寫入與 20 支限制
# =================================================================
def run_daily_sync():
    try:
        client = init_gspread()
        sh = client.open("users")
        ws_p = sh.worksheet("predictions")
        ws_w = sh.worksheet("watchlist")
        
        tw_tz = pytz.timezone('Asia/Taipei')
        today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")
        
        # 讀取 Watchlist
        watchlist = [str(r['symbol']).strip() for r in ws_w.get_all_records() if r.get('symbol')]
        
        # [個人化提醒] 20支上限
        if len(watchlist) > 20:
            print(f"🚨 提醒：目前自選股共 {len(watchlist)} 支，已超過您設定的 20 支上限！")

        for symbol in watchlist:
            df, f_id = fetch_comprehensive_data(symbol)
            if df is None: continue
            
            # 執行 AI 大腦
            p_close, r_low, r_high, s_data = perform_strategic_ai_engine(df)
            
            # 構建寫入列 (A-Y 欄)
            # A:Date, B:Symbol, C:Pred, D:Low, E:High, F:Actual, G-L:Buy, M-R:Sell, S-X:Resist, Y:Error
            upload_row = [
                today_str, f_id, p_close, r_low, r_high, "待收盤更新"
            ] + s_data + [""]
            
            ws_p.append_row(upload_row)
            print(f"✅ {f_id} 30天全週期數據已記錄")
            time.sleep(1) # 避免 Google API 頻率過高

    except Exception as e:
        print(f"💥 腳本執行異常: {e}")

if __name__ == "__main__":
    run_daily_sync()
