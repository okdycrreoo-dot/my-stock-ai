import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Google Sheets 永久資料庫連接 ---
# 在 Streamlit Cloud 的 Secrets 中設定你的試算表網址
conn = st.connection("gsheets", type=GSheetsConnection)

def get_user_data():
    # 讀取現有帳號，如果表是空的則回傳空 Dataframe
    try:
        return conn.read(worksheet="Sheet1", ttl=0)
    except:
        return pd.DataFrame(columns=["username", "password"])

def save_user_data(df):
    # 將更新後的名單寫回 Google Sheets
    conn.update(worksheet="Sheet1", data=df)

# --- 登入與註冊介面 ---
def auth_page():
    st.sidebar.title("🔐 永久帳號系統")
    auth_mode = st.sidebar.radio("操作項目", ["登入", "新用戶註冊"])
    
    user_input = st.sidebar.text_input("帳號")
    pass_input = st.sidebar.text_input("密碼", type="password")

    df_users = get_user_data()

    if auth_mode == "新用戶註冊":
        if st.sidebar.button("確認註冊"):
            if user_input in df_users["username"].values:
                st.sidebar.error("此帳號已被註冊！")
            elif user_input and pass_input:
                new_user = pd.DataFrame([{"username": user_input, "password": pass_input}])
                updated_df = pd.concat([df_users, new_user], ignore_index=True)
                save_user_data(updated_df)
                st.sidebar.success("帳號已永久儲存！請切換至登入")
            else:
                st.sidebar.warning("請填寫完整資訊")
                
    else: # 登入模式
        if st.sidebar.button("立即進入系統"):
            # 檢查帳號密碼是否匹配
            user_record = df_users[df_users["username"] == user_input]
            if not user_record.empty and str(user_record.iloc[0]["password"]) == pass_input:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = user_input
                st.rerun()
            else:
                st.sidebar.error("帳號或密碼錯誤")

# --- (下方保留之前的 LSTM 模型與 UI 代碼) ---
# ... [與前次代碼相同] ...
