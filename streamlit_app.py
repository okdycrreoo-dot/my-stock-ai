import streamlit as st
from streamlit_gsheets import GSheetsConnection

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# 1. Streamlit 基本設定（必須第一行）
# --------------------------------------------------
st.set_page_config(
    page_title="AI 股價預測系統",
    layout="wide"
)

# --------------------------------------------------
# 2. Google Sheets 連線
# --------------------------------------------------
def get_connection():
    try:
        return st.connection(
            "gsheets",
            type=GSheetsConnection
        )
    except Exception as e:
        st.error(f"資料庫連線失敗，請檢查 Secrets 設定。\n錯誤訊息：{e}")
        st.stop()

def get_user_data(conn):
    try:
        df = conn.read(ttl=0)
        if "username" not in df.columns:
            return pd.DataFrame(columns=["username", "password"])
        return df.dropna(subset=["username"])
    except Exception:
        return pd.DataFrame(columns=["username", "password"])

# --------------------------------------------------
# 3. LSTM 預測模型
# --------------------------------------------------
def lstm_predict(df, days_to_predict, epochs):
    data = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    lookback = 60
    if len(scaled_data) < lookback:
        return "資料不足（至少 60 天）"

    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i - lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train = np.array(x_train).reshape(-1, lookback, 1)
    y_train = np.array(y_train)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        verbose=0
    )

    temp = scaled_data[-lookback:].reshape(1, lookback, 1)
    future = []

    for _ in range(days_to_predict):
        pred = model.predict(temp, verbose=0)
        future.append(pred[0, 0])
        temp = np.append(temp[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    result = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    )

    return round(float(result[-1][0]), 2)

# --------------------------------------------------
# 4. 主程式
# --------------------------------------------------
conn = get_connection()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================= 未登入 =================
if not st.session_state.logged_in:
    st.title("🚀 AI 股價深度學習預測系統")
    st.info("請先登入或註冊帳號")

    st.sidebar.title("🔐 會員系統")
    mode = st.sidebar.radio("操作模式", ["登入", "註冊"])

    username = st.sidebar.text_input("帳號")
    password = st.sidebar.text_input("密碼", type="password")

    users_df = get_user_data(conn)

    if mode == "註冊":
        if st.sidebar.button("註冊"):
            if username and password and username not in users_df["username"].astype(str).values:
                new_user = pd.DataFrame([{
                    "username": username,
                    "password": password
                }])
                conn.update(data=pd.concat([users_df, new_user], ignore_index=True))
                st.sidebar.success("註冊成功，請登入")
            else:
                st.sidebar.error("帳號已存在或欄位空白")

    else:
        if st.sidebar.button("登入"):
            row = users_df[users_df["username"].astype(str) == username]
            if not row.empty and str(row.iloc[0]["password"]) == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.sidebar.error("帳號或密碼錯誤")

# ================= 已登入 =================
else:
    st.title(f"📊 預測控制台（使用者：{st.session_state.user}）")

    if st.sidebar.button("登出"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 參數設定")

    symbol = st.sidebar.text_input("股票代號", "2330.TW")

    epochs = st.sidebar.select_slider(
        "訓練輪數 (Epochs)",
        options=[1, 5, 10, 20],
        value=5
    )

    periods = st.sidebar.multiselect(
        "預測期間",
        ["明日", "1週", "1個月"],
        default=["明日"]
    )

    if st.sidebar.button("開始 AI 預測"):
        if not periods:
            st.warning("請選擇至少一個預測期間")
        else:
            with st.spinner("AI 模型訓練中，請稍候..."):
                df = yf.download(symbol, period="2y", progress=False)

                if df.empty:
                    st.error("查無股票資料，請確認代號")
                else:
                    st.subheader(f"{symbol} 近兩年收盤價")
                    st.line_chart(df["Close"])

                    mapping = {"明日": 1, "1週": 5, "1個月": 22}
                    cols = st.columns(len(periods))

                    for i, p in enumerate(periods):
                        price = lstm_predict(df, mapping[p], epochs)
                        with cols[i]:
                            st.metric(f"{p} 預測價", f"${price}")
