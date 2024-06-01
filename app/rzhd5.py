#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных
file_path = r'C:\Users\kmarz\Downloads\dataset (2).xlsx'
data = pd.ExcelFile(file_path)
sheet_1 = data.parse('22.05.2024')


# Очистка данных
sheet_1_cleaned = sheet_1.dropna(subset=['Номерной знак ТС', 'Данные путевых листов, пробег', 'Данные телематики, пробег'])
sheet_1_cleaned['дата путевого листа'] = pd.to_datetime(sheet_1_cleaned['дата путевого листа'])
sheet_1_cleaned['Дата числовая'] = sheet_1_cleaned['дата путевого листа'].map(pd.Timestamp.toordinal)

# Функция для расчета коэффициентов и рейтинга эффективности
def calculate_coefficients_and_rating(df):
    df = df.copy()
    
    # Коэффициенты для "Путевые листы"
    def calc_putevye_listy_coef(row):
        probeg_diff = abs(row['Данные путевых листов, пробег'] - row['Данные телематики, пробег']) #/ row['Данные путевых листов, пробег']
        if probeg_diff > 0.20:
            return 0.4
        elif probeg_diff > 0.10:
            return 0.3
        elif probeg_diff > 0.05:
            return 0.2
        else:
            return 1.0

    # Коэффициенты для "Соответствие целевой структуре"
    # Коэффициенты для "Соответствие целевой структуре"
#    def calc_soe_structura_coef(row):
#        struct_diff = abs(row['Соответствие целевой структуре'] - row['Целевая структура']) / row['Соответствие целевой структуре']
#        if struct_diff > 0.20:
#            return 0.4
#        elif struct_diff > 0.10:
#            return 0.3
#        elif struct_diff > 0.05:
#            return 0.2
#        else:
#            return 1.0

    # Коэффициенты для "Штрафы"
    def calc_shtrafy_coef(row):
        if row['Штрафы'] == 0:
            return 1.0
#        shtraf_diff = row['Штрафы'] / row['Штрафы целевые']
#        if shtraf_diff > 0.25:
#            return 0.3
#        elif shtraf_diff > 0.15:
#            return 0.2
#        elif shtraf_diff > 0.05:
#            return 0.1
        else:
            return 0.1

    # Коэффициенты для "Манера вождения"
    def calc_manera_vozhdeniya_coef(row):
        if row['манера вождения'] == 0:
            return 0.0
        manera_diff = (6 - row['манера вождения']) / 6
        if manera_diff > 0.25:
            return 0.3
        elif manera_diff > 0.15:
            return 0.2
        elif manera_diff > 0.10:
            return 0.1
        else:
            return 1.0

    # Применение функций для расчета коэффициентов
    df['Коэффициент Путевые листы'] = df.apply(calc_putevye_listy_coef, axis=1)
#    df['Коэффициент Соответствие целевой структуре'] = df.apply(calc_soe_structura_coef, axis=1)
    df['Коэффициент Штрафы'] = df.apply(calc_shtrafy_coef, axis=1)
    df['Коэффициент Манера вождения'] = df.apply(calc_manera_vozhdeniya_coef, axis=1)

    # Весовые коэффициенты для расчета рейтинга
    weights = {
        'Путевые листы': 0.4,
        'Соответствие целевой структуре': 0.3,
        'Штрафы': 0.15,
        'Манера вождения': 0.15
    }

    # Расчет рейтинга эффективности
    df['Рейтинг эффективности'] = (
        df['Данные путевых листов, пробег'] * df['Коэффициент Путевые листы'] * weights['Путевые листы'] +
#        df['Соответствие целевой структуре'] * df['Коэффициент Соответствие целевой структуре'] * weights['Соответствие целевой структуре'] +
        df['Штрафы'] * df['Коэффициент Штрафы'] * weights['Штрафы'] +
        df['манера вождения'] * df['Коэффициент Манера вождения'] * weights['Манера вождения']
    )
    
    return df

# Применение расчета рейтинга
sheet_1_cleaned = calculate_coefficients_and_rating(sheet_1_cleaned)

# Прогнозирование рейтинга эффективности
def predict_rating(df, category):
    X = df[['Дата числовая']]
    y = df['Рейтинг эффективности']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Построение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Прогнозирование на тестовых данных
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

# Функция для отображения данных и графиков
def show_data():
    st.title("Анализ данных автопарка")

    st.sidebar.header("Фильтры")
    
    # Фильтр по дате
    min_date = sheet_1_cleaned['дата путевого листа'].min()
    max_date = sheet_1_cleaned['дата путевого листа'].max()
    date_range = st.sidebar.date_input("Выберите диапазон дат", [min_date, max_date])
    
    # Фильтр по подразделению
    subdivisions = sheet_1_cleaned['Наименование структурного подразделения'].unique()
    selected_subdivisions = st.sidebar.multiselect("Выберите подразделения", subdivisions, default=subdivisions)

    # Фильтр по категории прогнозирования
    category = st.sidebar.selectbox("Выберите категорию прогнозирования", ["Наименование полигона", "Номерной знак ТС", "Тип закрепления"])
    
    # Применение фильтров
    filtered_data = sheet_1_cleaned[
        (sheet_1_cleaned['дата путевого листа'] >= pd.to_datetime(date_range[0])) &
        (sheet_1_cleaned['дата путевого листа'] <= pd.to_datetime(date_range[1])) &
        (sheet_1_cleaned['Наименование структурного подразделения'].isin(selected_subdivisions))
    ]

    st.header("Пробег и телематика")
    st.write("Средний пробег по путевым листам: {:.2f} км".format(filtered_data['Данные путевых листов, пробег'].mean()))
    st.write("Средний пробег по данным телематики: {:.2f} км".format(filtered_data['Данные телематики, пробег'].mean()))
#    st.write("Средняя разница пробега: {:.2f} км".format(filtered_data['Пробег разница'].mean()))

    st.header("Штрафы и манера вождения")
    st.write("Среднее количество штрафов: {:.2f}".format(filtered_data['Штрафы'].mean()))
    st.write("Средняя оценка манеры вождения: {:.2f}".format(filtered_data['манера вождения'].mean()))

    st.header("Рейтинг эффективности использования транспортных средств")
    st.write("Средний рейтинг эффективности: {:.2f}".format(filtered_data['Рейтинг эффективности'].mean()))

    st.header("Дополнительные показатели")
    # Предположим, что данные о средней скорости, времени в пути и количестве поездок находятся в соответствующих столбцах
    if 'Средняя скорость' in filtered_data.columns:
        st.write("Средняя скорость: {:.2f} км/ч".format(filtered_data['Средняя скорость'].mean()))
    
    if 'Время в пути' in filtered_data.columns:
        st.write("Среднее время в пути: {:.2f} часов".format(filtered_data['Время в пути'].mean()))
    
    if 'Количество поездок' in filtered_data.columns:
        st.write("Среднее количество поездок: {:.2f}".format(filtered_data['Количество поездок'].mean()))

    st.header("Графики")

    fig, ax = plt.subplots()
    ax.hist(filtered_data['Данные путевых листов, пробег'], bins=30, alpha=0.5, label='Путевые листы')
    ax.hist(filtered_data['Данные телематики, пробег'], bins=30, alpha=0.5, label='Телематика')
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # График штрафов
    st.write("График штрафов")
    fig, ax = plt.subplots()
    ax.hist(filtered_data['Штрафы'], bins=30, alpha=0.5, label='Штрафы')
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # График манеры вождения
    st.write("График манеры вождения")
    fig, ax = plt.subplots()
    ax.hist(filtered_data['манера вождения'], bins=30, alpha=0.5, label='Манера вождения')
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # График средней скорости
    if 'Средняя скорость' in filtered_data.columns:
        st.write("График средней скорости")
        fig, ax = plt.subplots()
        ax.hist(filtered_data['Средняя скорость'], bins=30, alpha=0.5, label='Средняя скорость')
        ax.legend(loc='upper right')
        st.pyplot(fig)
    
    # График времени в пути
    if 'Время в пути' in filtered_data.columns:
        st.write("График времени в пути")
        fig, ax = plt.subplots()
        ax.hist(filtered_data['Время в пути'], bins=30, alpha=0.5, label='Время в пути')
        ax.legend(loc='upper right')
        st.pyplot(fig)
    
    # График количества поездок
    if 'Количество поездок' in filtered_data.columns:
        st.write("График количества поездок")
        fig, ax = plt.subplots()
        ax.hist(filtered_data['Количество поездок'], bins=30, alpha=0.5, label='Количество поездок')
        ax.legend(loc='upper right')
        st.pyplot(fig)

# Функция для отображения прогноза рейтинга эффективности
def show_forecast():
    st.title("Прогнозирование рейтинга эффективности использования транспортных средств")

    st.sidebar.header("Фильтры")

    # Фильтр по категории прогнозирования
    category = st.sidebar.selectbox("Выберите категорию прогнозирования", ["Наименование полигона", "Номерной знак ТС", "Тип закрепления"])

    # Применение фильтров
    if category == "Наименование полигона":
        selected_category = st.sidebar.selectbox("Выберите полигон", sheet_1_cleaned['Наименование полигона'].unique())
        data_for_forecast = sheet_1_cleaned[sheet_1_cleaned['Наименование полигона'] == selected_category]
    elif category == "Номерной знак ТС":
        selected_category = st.sidebar.selectbox("Выберите номерной знак ТС", sheet_1_cleaned['Номерной знак ТС'].unique())
        data_for_forecast = sheet_1_cleaned[sheet_1_cleaned['Номерной знак ТС'] == selected_category]
    elif category == "Тип закрепления":
        selected_category = st.sidebar.selectbox("Выберите тип закрепления", sheet_1_cleaned['Тип закрепления'].unique())
        data_for_forecast = sheet_1_cleaned[sheet_1_cleaned['Тип закрепления'] == selected_category]

    # Прогнозирование рейтинга
    model, mse = predict_rating(data_for_forecast, category)

    st.header("Данные и модель")
    st.write(f"Среднеквадратичная ошибка модели: {mse:.2f}")

    # График фактических и предсказанных значений
    X = data_for_forecast[['Дата числовая']]
    y = data_for_forecast['Рейтинг эффективности']
    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Фактические значения')
    ax.scatter(X, y_pred, color='red', label='Предсказанные значения')
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # Прогнозирование на будущее
    future_dates = pd.date_range(start=sheet_1_cleaned['дата путевого листа'].max(), periods=30).map(pd.Timestamp.toordinal)
    future_predictions = model.predict(np.array(future_dates).reshape(-1, 1))

    # График прогноза на будущее
    fig, ax = plt.subplots()
    ax.plot(pd.date_range(start=sheet_1_cleaned['дата путевого листа'].max(), periods=30), future_predictions, color='green', label='Прогноз на будущее')
    ax.legend(loc='upper right')
    st.pyplot(fig)

# Запуск приложения
if __name__ == "__main__":
    page = st.sidebar.selectbox("Выберите страницу", ["Анализ данных", "Прогнозирование"])
    
    if page == "Анализ данных":
        show_data()
    elif page == "Прогнозирование":
        show_forecast()


# In[ ]:




