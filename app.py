import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import os

st.title("Кластеризация покупателей")

# Пути к файлам
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'
encoders_path = 'encoders.pkl'

# Признаки
numeric_features = ['Purchase_Frequency', 'Average_Order_Value', 
                    'Time_Between_Purchases', 'Churn_Probability', 'Lifetime_Value']

categorical_features = ['Season', 'Preferred_Purchase_Times', 'Retention_Strategy']

# загрузка обученной модели
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoders = joblib.load(encoders_path)

# ИНТЕРФЕЙС
st.sidebar.header("Данные клиента")

# Числовые признаки
st.sidebar.subheader("Числовые данные")
freq = st.sidebar.number_input("Частота покупок (в месяц)", min_value=1, max_value=50, value=10)
avg_order = st.sidebar.number_input("Средний чек (в долларах)", min_value=10, max_value=500, value=100)
time_between = st.sidebar.number_input("Дней между покупками", min_value=1, max_value=365, value=30)
churn = st.sidebar.slider("Вероятность оттока", 0.0, 1.0, 0.5)
ltv = st.sidebar.number_input("Lifetime Value ($)", min_value=100, max_value=10000, value=2000)

# Категориальные признаки
st.sidebar.subheader("Категориальные данные")

# Получаем возможные значения из обученных энкодеров
season_options = list(encoders['Season'].classes_)
preferred_time_options = list(encoders['Preferred_Purchase_Times'].classes_)
strategy_options = list(encoders['Retention_Strategy'].classes_)

season = st.sidebar.selectbox("Сезон", season_options)
preferred_time = st.sidebar.selectbox("Время покупок", preferred_time_options)
strategy = st.sidebar.selectbox("Стратегия удержания", strategy_options)

if st.sidebar.button("Определить кластер"):
    # Кодируем категориальные
    season_encoded = encoders['Season'].transform([season])[0]
    time_encoded = encoders['Preferred_Purchase_Times'].transform([preferred_time])[0]
    strategy_encoded = encoders['Retention_Strategy'].transform([strategy])[0]
    
    # Собираем все признаки
    features = np.array([[freq, avg_order, time_between, churn, ltv, 
                          season_encoded, time_encoded, strategy_encoded]])
    
    # Масштабируем
    features_scaled = scaler.transform(features)
    
    # Предсказываем
    cluster = model.predict(features_scaled)[0]
    
    # Выводим результат
    st.subheader("Результат кластеризации:")
    
    if cluster == 0:
        st.success("Активный клиент")
        st.info("Рекомендация: Программа лояльности, персональные предложения")
    else:
        st.warning("Неактивный клиент")
        st.info("Рекомендация: Скидки, ремаркетинг, вовлекающие акции")