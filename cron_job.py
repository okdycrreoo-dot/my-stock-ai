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

# -----------------------------------------------------------------
# 1. 初始化與連線
# -----------------------------------------------------------------
def init_gspread():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("環境變數 GCP_SERVICE_ACCOUNT_JSON 缺失")
    info = json.loads(creds_json)
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    return gspread.authorize(creds)

# -----------------------------------------------------------------
# 2. 數據引擎 (確保獲取足夠長度以分析 30 日數據)
# -----------------------------------------------------------------
def fetch_comprehensive_data(symbol):
    raw_s = str(symbol).strip().upper()
    search_list = [raw_s] if (raw_s.endswith(".TW") or raw_s.endswith(".TWO")) else [f"{raw_s}.TW", f"{raw_s}.TWO"]

    for s in search_list:
        try:
            # 抓取 2 年數據確保技術指標與 MA30 穩定
            df = yf.download(s, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) > 40:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                return df, s
        except: continue
    return None, raw_s

# -----------------------------------------------------------------
# 3. AI 核心：30天戰略大腦 (包含買價、賣價、壓力價)
# -----------------------------------------------------------------
def perform_strategic_ai_engine(df, p_days=30):
    last = df.iloc[-1]
    curr_p = float(last['Close'])
    periods = [5, 10, 15, 20, 25, 30]
    
    # 算力參數設定 (對標原 AI 大腦)
    rets = df['Close'].pct_change().dropna()
    f_vol = rets.tail(20).std()  # 20日波動率
    
    # 計算乖離率 (以 MA20 為基準)
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    bias = (curr_p - ma20) / (ma20 + 1e-5)

    strategic_data = [] 
    buy_levels, sell_levels, resist_levels = [], [], []
    
    for p in periods:
        sub = df.tail(p)
        ma = sub['Close'].mean()
        std = sub['Close'].std()
        h_max = sub['High'].max()
        l_min = sub['Low'].min()
        
        # 1. 建議買價 (Support)：均線 - 1.5倍標準差 與 近期低點加權
        buy_p = (ma - (std * 1.5)) * 0.4 + l_min * 0.6
        # 2. 建議賣價 (Target)：趨勢擴張位
        sell_p = ma + (std * 1.3)
        # 3. 壓力價格 (Resistance)：該週期天花板
        resist_p = max(h_max, ma + (std * 2.1))
        
        buy_levels.append(round(buy_p, 2))
        sell_levels.append(round(sell_p, 2))
        resist_levels.append(round(resist_p, 2))
    
    strategic_data = buy_levels + sell_levels + resist_levels

    # 4. 蒙地卡羅模擬 (1000次路徑計算)
    np.random.seed(42)
    sim_results = []
    # 考慮趨勢偏移 (Drift) 與 乖離拉力 (Reversion)
    drift = rets.tail(10).mean() 
    
    for _ in range(1000):
        # 融入 30 天內的隨機波動噪音
        noise = np.random.normal(drift, f_vol * 1.7, 7) # 預測未來 7 天走勢
        path = [curr_p]
        for n in noise:
            # 加入乖離率修正拉力，避免模擬跑得太離譜
            reversion = bias * 0.05 
            next_p = path[-1] * (1 + n - reversion)
            path.append(next_p)
        sim_results.append(path[1:])
    
    pred_7d = np.mean(sim_results, axis=0)[0] # 明日預測
    std_7d = np.std([p[0] for p in sim_results])
    
    return round(pred_7d, 2), round(pred_7d - std_7d*1.5, 2), round(pred_7d + std_7d*1.5, 2), strategic_data

# -----------------------------------------------------------------
# 4. 自動化寫入邏輯
# -----------------------------------------------------------------
def run_daily_sync():
    try:
        client = init_gspread()
        sh = client.open("users")
        ws_p = sh.worksheet("predictions")
        ws_w = sh.worksheet("watchlist")
        
        tw_tz = pytz.timezone('Asia/Taipei')
        today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")
        
        # 檢查 Watchlist
        records = ws_w.get_all_records()
        watchlist = [str(r['symbol']).strip() for r in records if r.get('symbol')]
        
        if len(watchlist) > 20:
            print(f"⚠️ 注意：目前清單共 {len(watchlist)} 支，已超過 20 支限制！")

        for symbol in watchlist:
            df, f_id = fetch_comprehensive_data(symbol)
            if df is None: continue
            
            # 核心運算
            p_close, r_low, r_high, s_data = perform_strategic_ai_engine(df)
            
            # 準備 A-Y 欄數據
            # A-F: 基本 / G-L: Buy / M-R: Sell / S-X: Resist / Y: Error
            upload_row = [
                today_str, f_id, p_close, r_low, r_high, "待收盤更新"
            ] + s_data + [""]
            
            ws_p.append_row(upload_row)
            print(f"✅ {f_id} 分析完成 (含5-30D壓力買賣價)")
            time.sleep(1.5) # 緩衝避免 API 報錯

    except Exception as e:
        print(f"💥 異常錯誤: {e}")

if __name__ == "__main__":
    run_daily_sync()
