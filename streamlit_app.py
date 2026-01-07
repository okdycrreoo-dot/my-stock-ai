import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# 設定網頁標題
st.set_page_config(page_title="AI 股價深度學習預測", layout="wide")
st.title("📈 LSTM 股價深度學習預測系統")

# --- LSTM 核心邏輯 (與你之前成功執行的相同) ---
def lstm_predict(df, days_to_predict, user_epochs):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    prediction_days = 60
    
    if len(scaled_data) < prediction_days: return None
    
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
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
    return round(float(res[-1][0]), 2) # 這裡使用了你剛剛修正成功的 [0]

# --- 側邊欄設定 ---
st.sidebar.header("參數設定")
symbol = st.sidebar.text_input("輸入股票代號", "2330.TW")
user_epochs = st.sidebar.slider("訓練輪數 (Epochs)", 1, 50, 5)
periods = st.sidebar.multiselect(
    "選擇預測期間", 
    ["明日", "1週", "1個月", "半年", "一年"],
    default=["明日", "1週"]
)

if st.sidebar.button("開始 AI 分析"):
    with st.spinner('AI 正在學習歷史數據，請稍候...'):
        df = yf.download(symbol, period="2y", progress=False)
        if not df.empty:
            # 數據顯示
            st.subheader(f"{symbol} 歷史股價 (過去兩年)")
            st.line_chart(df['Close'])
            
            # 預測邏輯
            period_map = {"明日": 1, "1週": 5, "1個月": 22, "半年": 126, "一年": 252}
            results = {}
            for p in periods:
                days = period_map.get(p)
                results[p] = lstm_predict(df, days, user_epochs)
            
            # 顯示結果卡片
            cols = st.columns(len(results))
            for i, (p, val) in enumerate(results.items()):
                with cols[i]:
                    st.metric(label=f"{p} 預測價", value=f"${val}")
        else:
            st.error("找不到股票代號，請檢查輸入是否正確。")