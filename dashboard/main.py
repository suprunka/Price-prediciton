import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from dbConnection import *
from bokeh.models.glyphs import Quad
from bokeh.models import ColumnDataSource
import pandas as pd

housing_n = get_data()
valueOfSqM = 10.76
housing_n = housing_n.drop(["sqft_living15", "sqft_lot15", 'date', 'waterfront'], axis=1)
housing_n['floors'] = housing_n['floors'].str[1:-1]
housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
housing = housing_n.convert_objects(convert_numeric=True)
housing['sqm_living'] = round(housing['sqft_living']/valueOfSqM)
housing['sqm_lot'] = round(housing['sqft_lot']/valueOfSqM)
housing['sqm_above'] = round(housing['sqft_above']/valueOfSqM)
housing['sqm_basement'] = round(housing['sqft_basement']/valueOfSqM)
housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)

description = housing['price'].describe()
print(description)

arr_hist, edges = np.histogram(housing['price'])

prices = pd.DataFrame({'arr_': arr_hist,
                       'left': edges[:-1],
                       'right': edges[1:]})

p = figure(plot_height=600, plot_width=600,
           title='Histogram of price and condition', x_axis_label='Condition',
           y_axis_label='Price')

p.quad(bottom=0, top=prices['arr_'], left=prices['left'], right=prices['right'], fill_color='red',
       line_color='black')

show(p)
