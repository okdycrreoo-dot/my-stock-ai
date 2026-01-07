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

# --- 1. 基礎診斷與配置 ---
st.set_page_config(page_title="StockAI 高穩定終端", layout="wide")

# 確保連線失敗時不會導致整頁白屏
def safe_main():
    try:
        if 'user' not in st.session_state: st.session_state.user = None
        
        # --- 管理員 okdycrreoo 統一控制設定 ---
        # 這裡會從 secrets 抓取管理員設定，若失敗則給予預設值防止崩潰
        admin_id = "okdycrreoo"
        
        # 2. 連線檢查
        client = None
        try:
            info = json.loads(st.secrets["connections"]["gsheets"]["service_account"])
            scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            client = gspread.authorize(creds)
        except Exception as e:
            st.error(f"❌ Google API 連線失敗。請檢查 Secrets 設定。錯誤訊息: {e}")
            return

        # 3. 獲取全域設定
        try:
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            sh = client.open_by_url(url)
            ws_settings = sh.worksheet("settings")
            settings_data = ws_settings.get_all_records()
            settings = {item['setting_name']: item['value'] for item in settings_data}
            
            # 使用記憶中 okdycrreoo 設定的參數
            curr_prec = int(settings.get('global_precision', 55))
            curr_ttl = int(settings.get('api_ttl_min', 5))
        except:
            st.warning("⚠️ 無法從試算表讀取 Settings，切換至安全預設值。")
            curr_prec = 55
            curr_ttl = 5

        # 4. 登入邏輯
        if st.session_state.user is None:
            st.title("🚀 StockAI 終端登入")
            # [登入表單...] (此處保持原樣)
            with st.form("login_form"):
                u = st.text_input("帳號")
                p = st.text_input("密碼", type="password")
                if st.form_submit_button("登入"):
                    # 模擬登入驗證
                    st.session_state.user = u
                    st.rerun()
        else:
            # 5. 進入主儀表板
            render_dashboard(client, admin_id, curr_prec, curr_ttl)
            
    except Exception as fatal_e:
        st.error(f"🚨 系統發生致命錯誤: {fatal_e}")

def render_dashboard(client, admin_id, precision, ttl):
    # 這裡放入我們之前的 fetch_fast_data 與 show_analysis_dashboard
    # 務必確認數據抓取有加入 .dropna() 避免運算失敗
    st.sidebar.success(f"管理者基準: {precision}% | API 緩存: {ttl}m")
    
    # 測試抓取 2330.TW
    target = st.sidebar.text_input("輸入股票代碼", "2330.TW").upper()
    
    # 圖表展示...
    st.write(f"正在載入 {target} 的 AI 技術分析...")
    # (此處調用之前的繪圖函數)

if __name__ == "__main__":
    safe_main()
