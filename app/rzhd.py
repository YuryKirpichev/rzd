#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Загрузка данных
file_path = r'dataset.xlsx'
data = pd.ExcelFile(file_path)
sheet_1 = data.parse('22.05.2024')

# Очистка данных
sheet_1_cleaned = sheet_1.dropna(subset=['Номерной знак ТС', 'дата путевого листа', 'Данные путевых листов, пробег'])

# Преобразование даты в числовой формат
sheet_1_cleaned['дата путевого листа'] = pd.to_datetime(sheet_1_cleaned['дата путевого листа'])
sheet_1_cleaned['Дата числовая'] = sheet_1_cleaned['дата путевого листа'].map(pd.Timestamp.toordinal)

# Выбор признаков и целевой переменной
X = sheet_1_cleaned[['Дата числовая']]
y = sheet_1_cleaned['Данные путевых листов, пробег']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)

# Функция для отображения данных и графиков
def show_data():
    st.title("Прогнозирование пробега транспортных средств")

    st.sidebar.header("Фильтры")
    
    # Фильтр по дате
    min_date = sheet_1_cleaned['дата путевого листа'].min()
    max_date = sheet_1_cleaned['дата путевого листа'].max()
    date_range = st.sidebar.date_input("Выберите диапазон дат", [min_date, max_date])
    
    # Фильтр по подразделению
    subdivisions = sheet_1_cleaned['Наименование структурного подразделения'].unique()
    selected_subdivisions = st.sidebar.multiselect("Выберите подразделения", subdivisions, default=subdivisions)
    
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

    st.header("Дополнительные показатели")
    # Предположим, что данные о средней скорости, времени в пути и количестве поездок находятся в соответствующих столбцах
    if 'Средняя скорость' in filtered_data.columns:
        st.write("Средняя скорость: {:.2f} км/ч".format(filtered_data['Средняя скорость'].mean()))
    
    if 'Время в пути' in filtered_data.columns:
        st.write("Среднее время в пути: {:.2f} часов".format(filtered_data['Время в пути'].mean()))
    
    if 'Количество поездок' in filtered_data.columns:
        st.write("Среднее количество поездок: {:.2f}".format(filtered_data['Количество поездок'].mean()))

    st.header("Графики")

    # График пробега
    st.write("График пробега")
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

# Запуск приложения
if __name__ == "__main__":
    show_data()

