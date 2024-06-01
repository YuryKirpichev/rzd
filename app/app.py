import io
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

from flask import Flask
from flask_recaptcha import ReCaptcha

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

import dash_bootstrap_components as dbc

from dash import dash_table
from future.builtins import disabled

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import base64
from datetime import date

pd.set_option('display.max_rows', 500)

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
    
# Коэффициенты для "Штрафы"
def calc_shtrafy_coef(row):
    if row['Штрафы'] == 0:
        return 1.0
    else:
        return 0

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



def parse_contents(contents, filename, date):
    """ function to read uploaded table
    input content and file names
    output table in pandas format
    """

    # as the first step verifying recaptcha

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
            if '.csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif '.xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                # if neither xlsx or csv file is uploaded make a warning instead
                df = 'Загружен файл неправильного формате, пожалуйста, загрузите excel или csv файл'
    except Exception as e:
            print(e)
    return df
    


# _____CODE STARTS HERE_______
# set of external styles
external_stylesheets = [dbc.themes.BOOTSTRAP]

# creating a web server instance
server = Flask(__name__)

# creating app
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets,
                title='Комплексная оценка эффективности использования автопарка', )

# putting a "spell" for yandex metrics
with open('metrics.txt') as f:
    ya_metrics = f.read()

app.index_string = ya_metrics



# creating pop up with help and info information
with open('info.txt') as f:
    info_text_rus = f.read()


pop_up = html.Div(
    [
        dbc.Button("О нас",
                   id="open",),
        dbc.Modal(html.Div(
            [
                html.Img(src='https://telemedai.ru/media/images/ms.original.png',
                         style={
                             'display': 'inline-block',
                             'width': 100,
                             'padding': 4
                         }),
                dbc.ModalHeader("Комплексная оценка эффективности использования автопарка",
                                style={'padding': 4}, id='pop_up_title'),
                dbc.ModalBody([
                    dcc.Markdown(info_text_rus, id='info_text')
                ]),
                dbc.ModalFooter(
                    dbc.Button("Закрыть", id="close", className="ml-auto")
                ),
            ]),
            id="modal",
        ),
    ], style={'display': 'inline-block', }
)

# creating the app layout
app.layout = html.Div([

    dcc.Store(id='memory-output'),
    html.Div([
        dbc.Row([
            dbc.Col(
                # header image
                html.Div([
                    html.Img(src='https://telemedai.ru/media/images/ms.original.png',
                         style={'display': 'inline-block', 'width': 100, 'padding': 10}),
                    html.P('CENTER FOR DIAGNOSTICS & TELEMEDICINE', 
                           style={
                                'display': 'inline-block',
                                'textAlign': 'right',
                                'color': 'white',
                                'width': 250}),]),
                width=3
            ),
            dbc.Col(html.Div([html.H1('Комплексная оценка эффективности использования автопарка')],
                             style={'textAlign': 'left', 'color': 'white'}),
                    width=5
                    ),
            dbc.Col(html.Div([
                pop_up,
            ], style={'textAlign': 'right'}),
                width=3
            ),
        ], style={'margin-bottom': 5, 'margin-left': 100}),
        html.Br(),

        html.H1('',
                style={
                    'display': 'inline-block',
                    'padding': 10,
                    'textAlign': 'right',
                    'margin-top': 20,
                    'margin-left': 100,
                    'color': 'white'},
                id='title'),

        # pop_up with information

    ], style={'backgroundColor': '#061c42', 'padding': 10}),

    html.Br(style={'height': 2}),
    
    # filters
    html.Div([
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=date(2020, 1, 1),
            max_date_allowed=date(2025, 1, 1),
            initial_visible_month=date(2024, 8, 5),
            start_date = date(2024,5,1),
            end_date=date(2024, 6, 1)),
        
        html.Strong('Наименование полигона', style = {'padding': 10}),
        dcc.Dropdown(id='car_group', options = [], style ={ 'width': '300px',  'display': 'inline-block'}),
        html.Strong('Номер машины',  style = {'padding': 10}),
        dcc.Dropdown(id='car_number',  options = [], style ={ 'width': '300px', 'display': 'inline-block'})
    ], style = {'padding': 10, 'display': 'inline-block'}),

    # uploading field
    dcc.Upload(
        id='upload-data',
        max_size=-1,
        children=html.Div(['Upload file'], style={"overflow": "hidden"}, id='upload_text'),
        style={
            'max-width': '98%',
            'height': '100px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'padding': 10,
            'cursor': 'pointer'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    # place for the output
    html.Div(id='output-data-upload', style={'padding': 10, 'margin-left': 90}),
])


# main callback

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               
               ],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents,
                  list_of_names,
                  list_of_dates, ):
    try:
        if list_of_contents is not None:
            df = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)][0]
            name_df = list_of_names[0]
            
            # Очистка данных
            df.loc[df['Наименование полигона'].isnull(), 'grouped'] = False
            df.loc[df['Наименование полигона'].notnull(), 'grouped'] = True

            #filling the values with previous
            df = df.fillna(method='ffill')

            # Применение функций для расчета коэффициентов
            df['Коэффициент Путевые листы'] = df.apply(calc_putevye_listy_coef, axis=1)
            # df['Коэффициент Соответствие целевой структуре'] = df.apply(calc_soe_structura_coef, axis=1)
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
            #       df['Соответствие целевой структуре'] * df['Коэффициент Соответствие целевой структуре'] * weights['Соответствие целевой структуре'] +
                    df['Штрафы'] * df['Коэффициент Штрафы'] * weights['Штрафы'] +
                    df['манера вождения'] * df['Коэффициент Манера вождения'] * weights['Манера вождения']
    )

            #removing the missing values
            df_clean = df[df['grouped'] != True]


            # Преобразование даты в числовой формат
            df_clean['дата путевого листа'] = pd.to_datetime(df_clean['дата путевого листа'])
            df_clean['Дата числовая'] = df_clean['дата путевого листа'].map(pd.Timestamp.toordinal)
            
            #фильтрование данных
            filtered_data = df_clean
            # Выбор признаков и целевой переменной
            X = df_clean[['Дата числовая']]
            y = df_clean['Данные путевых листов, пробег']

            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Построение модели линейной регрессии
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Прогнозирование на тестовых данных
            y_pred = model.predict(X_test)

            # Оценка модели
            mse = mean_squared_error(y_test, y_pred)

            children = [html.Span(f'Название загруженного файла: {name_df}')]
            
            children += html.Div([
                dash_table.DataTable(
                data=df_clean.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                export_format="xlsx",
                # editable=True,
                style_table={'display': 'inline-block'}),
                ], style={'display': 'inline-block', "width": '90%', 'overflowY': 'auto','height': '600px'}),

            #Пробег и телематика
            mean_pass_list = filtered_data['Данные путевых листов, пробег'].mean()
            mean_pass_telematika = filtered_data['Данные телематики, пробег'].mean()
            children += [
                html.H3('Пробег и телематика'), html.Br(),
                html.P(f'Средний пробег по путевым листам: {mean_pass_list:.2f} км'),
                html.P(f'Средний пробег по данным телематики: {mean_pass_telematika:.2f} км')
            ]

            #Штрафы и манера вождения

            penalty =  filtered_data['Штрафы'].mean()
            driving_style = filtered_data['манера вождения'].mean()

            children += [
                html.H3('Штрафы и манера вождения'),
                html.P(f'Среднее количество штрафов: {penalty:.2f}'),
                html.P(f'Средняя оценка манеры вождения: {driving_style:.2f}')
            ]

            #Рейтинг
            fig =  go.Figure()
            fig.update_layout(title="Рейтинг",)
            fig.add_trace(go.Histogram(
                 x = filtered_data['Рейтинг эффективности'],
                 name = 'Рейтинг',
                 ))

            children += [dcc.Graph(figure=fig,)]

            # График пробега
            fig =  go.Figure()
            fig.update_layout(title="График пробега",)
            fig.add_trace(go.Histogram(
                 x = filtered_data['Данные путевых листов, пробег'],
                 name = 'Путевые листы',
                 ))
            fig.add_trace(go.Histogram(
                 x = filtered_data['Данные телематики, пробег'],
                 name = 'Телематика',
                 
            ))
            children += [dcc.Graph(figure=fig,)]

            # График штрафов
            fig = go.Figure()
            fig.update_layout(title="График штрафов",)
            fig.add_trace(go.Histogram(
                 x = filtered_data['Штрафы'],
                 name = 'Штрафы',
                 ))
            children += [dcc.Graph(figure=fig,)]

            # График манеры вождения
            fig = go.Figure()
            fig.update_layout(title="График манеры вождения",)
            fig.add_trace(go.Histogram(
                 x = filtered_data['манера вождения'],
                 name = 'манера вождения',
                 ))
            children += [dcc.Graph(figure=fig,)]

        else:
            children = [html.Span('Загрузите файл для анализа.'), ]    

        return children
    except Exception as e:

        print(e)
        children = [html.Span(f'Что-то пошло не так, попробуйте еще раз. {e}'), ]
        return children


# call back for pop up window
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open




if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)

