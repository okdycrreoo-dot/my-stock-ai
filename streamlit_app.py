import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 必須是第一個 Streamlit 指令
st.set_page_config(page_title="AI 股價預測系統", layout="wide")

# 2. 資料庫連線
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("Secrets 設定有誤，請確認。")
    st.stop()

def get_user_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        return df.dropna(subset=["username"])
    except:
        return pd.DataFrame(columns=["username", "password"])

# 3. LSTM 運算函數
def lstm_predict(df, days_to_predict, user_epochs):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    prediction_days = 60
    
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

# 4. 主程式邏輯
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🚀 AI 股價預測系統")
    st.info("請從左側登入或註冊以使用功能。")
    
    mode = st.sidebar.radio("帳號管理", ["登入", "註冊帳號"])
    u = st.sidebar.text_input("帳號")
    p = st.sidebar.text_input("密碼", type="password")
    
    df_users = get_user_data()
    
    if mode == "註冊帳號" and st.sidebar.button("確認註冊"):
        if u and p and u not in df_users["username"].values:
            new_data = pd.concat([df_users, pd.DataFrame([{"username": u, "password": p}])], ignore_index=True)
            conn.update(worksheet="Sheet1", data=new_data)
            st.sidebar.success("註冊成功！現在請切換到登入模式。")
        else:
            st.sidebar.error("帳號已存在或輸入空白。")
            
    if mode == "登入" and st.sidebar.button("登入系統"):
        user_record = df_users[df_users["username"] == u]
        if not user_record.empty and str(user_record.iloc[0]["password"]) == p:
            st.session_state['logged_in'] = True
            st.session_state['user'] = u
            st.rerun()
        else:
            st.sidebar.error("帳號或密碼錯誤。")
else:
    st.title(f"📊 歡迎使用, {st.session_state['user']}!")
    
    symbol = st.sidebar.text_input("輸入股票代號 (如: 2330.TW)", "2330.TW")
    user_epochs = st.sidebar.select_slider("訓練輪數 (Epochs)", options=[1, 5, 10], value=1)
    st.sidebar.warning("注意：選取多個期間會大幅增加運算時間。")
    
    periods = st.sidebar.multiselect("預測期間", ["明日", "1週", "1個月"], default=["明日"])

    if st.sidebar.button("開始 AI 運算"):
        with st.spinner('運算中...這可能需要一分鐘...'):
            df = yf.download(symbol, period="2y", progress=False)
            if not df.empty:
                st.line_chart(df['Close'])
                period_map = {"明日": 1, "1週": 5, "1個月": 22}
                cols = st.columns(len(periods))
                for i, p in enumerate(periods):
                    val = lstm_predict(df, period_map[p], user_epochs)
                    cols[i].metric(label=p, value=f"${val}")
            else:
                st.error("查無資料。")

    if st.sidebar.button("登出"):
        st.session_state['logged_in'] = False
        st.rerun()
