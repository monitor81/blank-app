import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Оценка стоимости дома", layout="centered")
st.title("🏠 Предсказание цены дома")
st.markdown("Введите параметры недвижимости — модель скажет её рыночную стоимость.")

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()

col1, col2 = st.columns(2)
with col1:
    square = st.number_input("Площадь (м²)", min_value=10.0, max_value=200.0, value=80.0, step=5.0)
with col2:
    rooms = st.slider("Количество комнат", min_value=1, max_value=5, value=3, step=1)

if st.button("Рассчитать цену", type="primary"):
    input_data = np.array([[square, rooms]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Прогнозируемая цена: **{prediction:.0f} тыс. у.е.**")
    st.caption("Модель: линейная регрессия, обучена на 800 примерах.")

with st.sidebar:
    st.header("О модели")
    st.write("Коэффициенты:")
    st.write(f"- за 1 м²: {model.coef_[0]:.2f} тыс. у.е.")
    st.write(f"- за 1 комнату: {model.coef_[1]:.2f} тыс. у.е.")
    st.write(f"- базовая цена: {model.intercept_:.0f} тыс. у.е.")
