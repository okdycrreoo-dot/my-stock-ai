import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# 1. Streamlit 基礎設定（必須第一個）
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
        # 使用 Streamlit 內建 gspread 連線器
        return st.connection("gsheets", type="gspread")
    except Exception as e:
        st.error(f"資料庫連線失敗，請檢查 Secrets 設定。\n錯誤訊息：{e}")
        st.stop()

def get_user_data(conn):
    try:
        # ttl=0 確保每次都讀取最新資料
        df = conn.read(ttl=0)
        if "username" not in df.columns:
            return pd.DataFrame(columns=["username", "password"])
        return df.dropna(subset=["username"])
    except Exception:
        return pd.DataFrame(columns=["username", "password"])

# --------------------------------------------------
# 3. LSTM 預測模型
# --------------------------------------------------
def lstm_predict(df, days_to_predict, user_epochs):
    data = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    prediction_days = 60
    if len(scaled_data) < prediction_days:
        return "數據不足（需至少 60 筆）"

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(
        x_train,
        (x_train.shape[0], x_train.shape[1], 1)
    )

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(prediction_days, 1)),
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
        epochs=user_epochs,
        verbose=0
    )

    # 預測未來
    temp_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    future_predictions = []

    for _ in range(days_to_predict):
        prediction = model.predict(temp_input, verbose=0)
        future_predictions.append(prediction[0, 0])
        temp_input = np.append(
            temp_input[:, 1:, :],
            prediction.reshape(1, 1, 1),
            axis=1
        )

    result = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
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
    st.info("👋 歡迎！請先註冊或登入以使用完整功能")

    st.sidebar.title("🔐 會員系統")
    mode = st.sidebar.radio(
        "請選擇操作",
        ["登入", "註冊帳號"],
        key="auth_mode"
    )

    username = st.sidebar.text_input("帳號", key="username")
    password = st.sidebar.text_input("密碼", type="password", key="password")

    df_users = get_user_data(conn)

    if mode == "註冊帳號":
        if st.sidebar.button("確認註冊"):
            if username and password and username not in df_users["username"].astype(str).values:
                new_user = pd.DataFrame([{
                    "username": username,
                    "password": password
                }])
                updated_df = pd.concat(
                    [df_users, new_user],
                    ignore_index=True
                )
                conn.update(data=updated_df)
                st.sidebar.success("註冊成功！請切換到登入模式")
            else:
                st.sidebar.error("帳號已存在或欄位空白")

    else:  # 登入
        if st.sidebar.button("登入"):
            user_row = df_users[df_users["username"].astype(str) == username]
            if not user_row.empty and str(user_row.iloc[0]["password"]) == password:
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
    st.sidebar.header("⚙️ 預測參數")

    symbol = st.sidebar.text_input(
        "股票代號（例：2330.TW / TSLA）",
        "2330.TW"
    )

    user_epochs = st.sidebar.select_slider(
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
            st.warning("請至少選擇一個預測期間")
        else:
            with st.spinner("AI 模型訓練中，請稍候..."):
                df = yf.download(
                    symbol,
                    period="2y",
                    progress=False
                )

                if df.empty:
                    st.error("查無股票資料，請確認代號是否正確")
                else:
                    st.subheader(f"📈 {symbol} 近兩年收盤價")
                    st.line_chart(df["Close"])

                    period_map = {
                        "明日": 1,
                        "1週": 5,
                        "1個月": 22
                    }

                    st.subheader("🤖 AI 預測結果")
                    cols = st.columns(len(periods))

                    for i, p in enumerate(periods):
                        price = lstm_predict(
                            df,
                            period_map[p],
                            user_epochs
                        )
                        with cols[i]:
                            st.metric(
                                label=f"{p} 預測價",
                                value=f"${price}"
                            )
