 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dill
Price_Patent_Reg = dill.load(open('features_created.pkd', 'rb'))

Price_Patent_Reg = pd.get_dummies(Price_Patent_Reg, drop_first = True)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(Price_Patent_Reg,
                                         test_size = 0.2,
                                         random_state = 1,)

from sklearn import base
class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, groupby_column, pipeline_factory):
        self.groupby_column = groupby_column
        self.pipeline_factory = pipeline_factory

    def fit(self, dataframe, label):
        self.drugs_dict = {}
        self.label = label
        self.coefs_dict = {}
        self.intercepts_dict = {}
        dataframe = pd.get_dummies(dataframe)
        for name, values in dataframe.groupby(self.groupby_column):
            y = values[label]
            X = values.drop(columns = [label, self.groupby_column], axis = 1)
            self.drugs_dict[name] = self.pipeline_factory().fit(X, y)
            self.coefs_dict[name] = self.drugs_dict[name].named_steps["lin_reg"].coef_
            self.intercepts_dict[name] = self.drugs_dict[name].named_steps["lin_reg"].intercept_
        return self

    def get_coefs(self):
        return self.coefs_dict

    def get_intercepts(self):
        return self.intercepts_dict

    def predict(self, test_data):
        price_pred_list = []

        for idx, row in test_data.iterrows():
            name = row[self.groupby_column]
            regression_coefs = self.drugs_dict[name]
            row = pd.DataFrame(row).T
            X = row.drop(columns = [self.label, self.groupby_column], axis = 1).values.reshape(1, -1)

            drug_price_pred = regression_coefs.predict(X)
            price_pred_list.append([name, drug_price_pred])
        return price_pred_list

def pipeline_factory():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    return Pipeline([('lin_reg', LinearRegression())])

lin_model = GroupbyEstimator('ndc', pipeline_factory).fit(train_data,'nadac_per_unit')


def format_data(dataframe, filename, test = False):
    dataframe.loc[:, 'ndc'] = dataframe.loc[:, 'ndc'].astype('int64')
    if test:
        dataframe.loc[:, ['effective_date_year', 'effective_date_month', 'effective_date_day']] = dataframe.loc[:, ['effective_date_year', 'effective_date_month', 'effective_date_day']].astype(str)
        dataframe.rename(columns = {'effective_date_year': 'year', 'effective_date_month': 'month', 'effective_date_day': 'day'}, inplace = True)
        dataframe.loc[:, 'date'] = pd.to_datetime(dataframe[['year', 'month', 'day']], format = '%Y-%m-%d')
        dataframe.rename({'year': 'effective_date_year', 'month': 'effective_date_month', 'day': 'effective_date_day'}, inplace = True)
        dataframe.loc[:, ['year', 'month', 'day']] = dataframe.loc[:, ['year', 'month', 'day']].astype(float).astype(int)
        dataframe.sort_values(['ndc', 'date'])
    else:
        dataframe.rename(columns = {'effective_date_year': 'year', 'effective_date_month': 'month', 'effective_date_day': 'day'}, inplace = True)
    dataframe.loc[:, 'year'] = dataframe.loc[:, 'year'].astype(int)
    dataframe.loc[:, 'month'] = dataframe.loc[:, 'month'].astype(int)
    dataframe.loc[:, 'day'] = dataframe.loc[:, 'day'].astype(int)
    dataframe.loc[:, 'nadac_per_unit'] = dataframe.loc[:, 'nadac_per_unit'].astype('float16')
    return dataframe

historical_data = format_data(train_data, 'historical_data', test = True)
prediction_data = format_data(test_data, 'pred_data')

from bokeh.io import curdoc, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, DataRange1d, HoverTool
from bokeh.plotting import figure

historical_data = historical_data.loc[:, ['ndc', 'date', 'nadac_per_unit']]
hist_temp = historical_data[historical_data.loc[:, 'ndc']==781593600].sort_values('date')
historical_source = ColumnDataSource(data = hist_temp)

import datetime as dt

date = dt.datetime.strptime('-'.join(('2025', '1', '1')), '%Y-%m-%d')
new_prediction_data = prediction_data[prediction_data.loc[:, 'ndc']==781593600]
new_prediction_data.loc[:, 'year'] = date.year
new_prediction_data.loc[:, 'month'] = date.month
new_prediction_data.loc[:, 'day'] = date.day
new_prediction_data = lin_model.predict(new_prediction_data)
new_prediction_data = pd.DataFrame(data = {'ndc':new_prediction_data[0][0], 'nadac_per_unit':new_prediction_data[0][1][0]}, index = [0])
new_prediction_data['date'] = pd.to_datetime(date, format='%Y-%m-%d')
new_prediction_data['ndc'] = new_prediction_data['ndc'].astype(float).astype('int64')
new_prediction_data['nadac_per_unit'] = new_prediction_data['nadac_per_unit'].astype('float16')
prediction_source = ColumnDataSource(data=new_prediction_data)
id_list = list(prediction_data['ndc'].astype(str))[0:10]
plot = figure(plot_height=800, plot_width=800, title='Medicine Price Prediction',
              x_axis_type = 'datetime',
              tools="crosshair, pan, reset, save, wheel_zoom")
plot.xaxis.axis_label = 'Time'
plot.yaxis.axis_label = 'Price ($)'
plot.axis.axis_label_text_font_style = 'bold'
plot.grid.grid_line_alpha = 0.8
plot.title.text_font_size = '16pt'
plot.x_range = DataRange1d(range_padding = .01)
plot.add_tools(HoverTool(tooltips=[('Price', '$@nadac_per_unit{0, 0.00}'), ('Date', '@date{%F}')], formatters = {'@date': 'datetime'}))

plot.line('date', 'nadac_per_unit', source=historical_source, legend='Historical Price')

plot.scatter('date', 'nadac_per_unit', source=prediction_source, color = "blue", size = 8, legend='Predicted Price')
id_select = Select(title='Select a Drug ID Number', value='781593600', options=id_list)

def update_data(attrname, old, new):
    curr_id = id_select.value
    new_historical = historical_data[historical_data.loc[:, 'ndc']==int(curr_id)]
    new_historical = new_historical.sort_values('date')

    new_prediction_data = prediction_data[prediction_data.loc[:, 'ndc']==int(curr_id)]
    date = dt.datetime.strptime('-'.join(('2025', '1', '1')), '%Y-%m-%d')
    new_prediction_data.loc[:, 'year'] = date.year
    new_prediction_data.loc[:, 'month'] = date.month
    new_prediction_data.loc[:, 'day'] = date.day
    new_prediction_data = lin_model.predict(new_prediction_data)
    new_prediction_data = pd.DataFrame(data = {'ndc':new_prediction_data[0][0], 'nadac_per_unit':new_prediction_data[0][1][0]}, index = [0])
    new_prediction_data['date'] = pd.to_datetime(date, format='%Y-%m-%d')
    new_prediction_data['ndc'] = new_prediction_data['ndc'].astype(float).astype('int64')

    historical_source.data = ColumnDataSource.from_df(new_historical)
    prediction_source.data = ColumnDataSource.from_df(new_prediction_data)


id_select.on_change('value', update_data)
inputs = column(id_select)

curdoc().add_root(row(inputs, plot, width = 1000))
curdoc().title = 'Medicine Price Predictor'
