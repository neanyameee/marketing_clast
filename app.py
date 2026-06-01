import streamlit as st
import numpy as np
import joblib # загрузка сохраненных моделей

# Настройка страницы
st.set_page_config(
    page_title="Кластеризация покупателей",
    layout="wide"
)

st.title(" Кластеризация покупателей")
st.markdown("Определите сегмент клиента для персонализированного маркетинга")

# Пути к файлам
model_path = 'model.pkl' # файл с обученной моделью KMeans
scaler_path = 'scaler.pkl' # файл с масштабатором
encoders_path = 'encoders.pkl' # файл с кодировщиками категорий

# Признаки
numeric_features = ['Purchase_Frequency', 'Average_Order_Value', 
                    'Time_Between_Purchases', 'Churn_Probability', 'Lifetime_Value']

categorical_features = ['Season', 'Preferred_Purchase_Times', 'Retention_Strategy']

# загрузка обученной модели
@st.cache_resource
def load_models():
    model = joblib.load(model_path) # загрузка KMeans
    scaler = joblib.load(scaler_path) # загрузка StandardScaler
    encoders = joblib.load(encoders_path) # загрузка LabelEncoder
    return model, scaler, encoders

model, scaler, encoders = load_models()

# Создаем две колонки для ввода данных
col1, col2 = st.columns(2)

# левая колонка Числовые данные
with col1:
    freq = st.number_input(" Частота покупок (в месяц)", min_value=1, max_value=50, value=10)
    avg_order = st.number_input(" Средний чек (в долларах)", min_value=10, max_value=500, value=100)
    time_between = st.number_input(" Дней между покупками", min_value=1, max_value=365, value=30)
    ltv = st.number_input(" Lifetime Value (в долларах)", min_value=100, max_value=10000, value=2000)

# правая колонка - Катеориальные данные
with col2:
    # Получаем возможные значения из обученных энкодеров
    season_options = list(encoders['Season'].classes_)
    preferred_time_options = list(encoders['Preferred_Purchase_Times'].classes_)
    strategy_options = list(encoders['Retention_Strategy'].classes_)
    
    season = st.selectbox(" Сезон", season_options)
    preferred_time = st.selectbox(" Время покупок", preferred_time_options)
    strategy = st.selectbox(" Стратегия удержания", strategy_options)
    churn = st.slider(" Вероятность оттока", 0.0, 1.0, 0.5)

# кнопка и результат
if st.button(" Определить кластер", use_container_width=True):
    # кодировка
    season_encoded = encoders['Season'].transform([season])[0]
    time_encoded = encoders['Preferred_Purchase_Times'].transform([preferred_time])[0]
    strategy_encoded = encoders['Retention_Strategy'].transform([strategy])[0]
        
    # сборка всех признаков
    features = np.array([[freq, avg_order, time_between, churn, ltv, 
                              season_encoded, time_encoded, strategy_encoded]])
    # масштабирование
    features_scaled = scaler.transform(features)
    # предсказание
    cluster = model.predict(features_scaled)[0]
        
    st.markdown("**Результат кластеризации**")  
    if cluster == 0:            
        st.success("Активный клиент")
        st.markdown("*Рекомендации:*")
        st.markdown("- Программа лояльности")
        st.markdown("- Персональные предложения")
        st.markdown("- Ранний доступ к новинкам")
    else:            
        st.warning("Неактивный клиент")
        st.markdown("*Рекомендации:*")
        st.markdown("- Скидки и промокоды")
        st.markdown("- Email-рассылка")
        st.markdown("- Ремаркетинг")