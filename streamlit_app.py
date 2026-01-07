# streamlit_app.py
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. 基礎設定 ---
st.set_page_config(page_title="AI 股價預測系統", layout="wide")

# --- 2. Google Sheets 連線 ---
def get_connection():
    try:
        gcp_sa = st.secrets["gcp_service_account"]
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_sa, scope)
        client = gspread.authorize(creds)
        sheet_url = "https://docs.google.com/spreadsheets/d/1EH1MlLyEWtk7t5mO0-nqtDFUoqN48AJ2YjTG2Jn6Rfc/edit#gid=0"
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.sheet1
        return worksheet
    except Exception as e:
        st.error(f"資料庫連線失敗，請檢查 Secrets 設定。錯誤: {e}")
        st.stop()

def get_user_data(worksheet):
    try:
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        if "username" not in df.columns:
            return pd.DataFrame(columns=["username", "password"])
        return df.dropna(subset=["username"])
    except Exception:
        return pd.DataFrame(columns=["username", "password"])

def update_user_data(worksheet, df):
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

# --- 3. LSTM 模型運算 ---
def lstm_predict(df, days_to_predict, user_epochs):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    prediction_days = 60
    if len(scaled_data) < prediction_days:
        return "數據量不足(需60天以上)"

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train = np.reshape(np.array(x_train), (len(x_train), prediction_days, 1))
    y_train = np.array(y_train)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(prediction_days, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=user_epochs, verbose=0)

    temp_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    future_preds = []
    for _ in range(days_to_predict):
        current_pred = model.predict(temp_input, verbose=0)
        future_preds.append(current_pred[0, 0])
        new_val = current_pred.reshape(1, 1, 1)
        temp_input = np.append(temp_input[:, 1:, :], new_val, axis=1)

    res = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    return round(float(res[-1][0]), 2)

# --- 4. 主程式介面邏輯 ---
worksheet = get_connection()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🚀 AI 股價深度學習預測系統")
    st.info("👋 歡迎！請先註冊或登入以開啟預測功能。")

    st.sidebar.title("🔐 會員管理")
    mode = st.sidebar.radio("請選擇操作", ["登入", "註冊帳號"], key="auth_mode")
    u = st.sidebar.text_input("帳號", key="user_input")
    p = st.sidebar.text_input("密碼", type="password", key="pass_input")

    df_users = get_user_data(worksheet)

    if mode == "註冊帳號":
        if st.sidebar.button("確認註冊並存入雲端", key="reg_btn"):
            if u and p and u not in df_users["username"].astype(str).values:
                new_row = pd.DataFrame([{"username": u, "password": p}])
                updated_df = pd.concat([df_users, new_row], ignore_index=True)
                update_user_data(worksheet, updated_df)
                st.sidebar.success("註冊成功！請切換到『登入』模式。")
            else:
                st.sidebar.error("帳號已存在或欄位空白。")

    elif mode == "登入":
        if st.sidebar.button("進入預測控制台", key="login_btn"):
            user_record = df_users[df_users["username"].astype(str) == u]
            if not user_record.empty and str(user_record.iloc[0]["password"]) == p:
                st.session_state['logged_in'] = True
                st.session_state['user'] = u
                st.rerun()
            else:
                st.sidebar.error("帳號或密碼錯誤。")
else:
    st.title(f"📊 預測中心 - 使用者：{st.session_state['user']}")

    if st.sidebar.button("登出帳號"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 運算參數設定")

    symbol = st.sidebar.text_input("股票代號 (例: 2330.TW, TSLA)", "2330.TW")
    st.sidebar.subheader("💡 效能警告")
    st.sidebar.caption("訓練輪數愈多，計算反饋愈慢；預測多個期間也會增加等待時間。")

    user_epochs = st.sidebar.select_slider("訓練輪數 (Epochs)", options=[1, 5, 10, 20], value=5)
    periods = st.sidebar.multiselect("選擇預測目標期間", ["明日", "1週", "1個月"], default=["明日"])

    if st.sidebar.button("啟動 AI 深度學習預測"):
        if not periods:
            st.warning("請至少選擇一個預測期間。")
        else:
            with st.spinner(f'AI 正在學習數據中，請稍候...'):
                df = yf.download(symbol, period="2y", progress=False)
                if not df.empty:
                    st.subheader(f"📈 {symbol} 過去兩年歷史走勢")
                    st.line_chart(df['Close'])

                    period_map = {"明日": 1, "1週": 5, "1個月": 22}
                    st.write("### AI 預測結果")
                    cols = st.columns(len(periods))

                    for i, p in enumerate(periods):
                        result = lstm_predict(df, period_map[p], user_epochs)
                        with cols[i]:
                            st.metric(label=f"{p} 預測價", value=f"${result}")
                else:
                    st.error("查無股票代號，請檢查輸入是否正確（台股請加 .TW）。")
