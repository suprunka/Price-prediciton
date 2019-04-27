import numpy as np
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure, show, gmap
from dbConnection import *
from bokeh.models.glyphs import Quad
from bokeh.models import ColumnDataSource, GMapOptions
import pandas as pd
from datetime import datetime
from math import pi
from bokeh.transform import cumsum
from bokeh.models import HoverTool
from bokeh.palettes import Category20c
from bokeh.tile_providers import CARTODBPOSITRON
import math
from ast import literal_eval


def merc(lat, lon):
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x / lon
        y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 +
                                                lat * (math.pi / 180.0) / 2.0)) * scale
        return (x, y)

housing_n = get_data()
valueOfSqM = 10.76
housing_n = housing_n.drop(["sqft_living15", "sqft_lot15", 'waterfront'], axis=1)
housing_n['floors'] = housing_n['floors'].str[1:-1]
housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
housing_n['date'] = housing_n['date'].apply(lambda x: x.replace(x, x[0:10]))
housing = housing_n.convert_objects(convert_numeric=True)
housing['sqm_living'] = round(housing['sqft_living']/valueOfSqM)
housing['sqm_lot'] = round(housing['sqft_lot']/valueOfSqM)
housing['sqm_above'] = round(housing['sqft_above']/valueOfSqM)
housing['sqm_basement'] = round(housing['sqft_basement']/valueOfSqM)
housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)
housing['date'] = housing['date'].apply(lambda x: x.replace(x, x[1:5]+"-"+x[5:7]+"-"+x[7:9]))
housing['date'] = housing['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())


# groupByGrade = housing[['price', 'grade']].groupby(['grade']).mean()
# legend_title = "House grades"
# groupByGrade['angle'] = groupByGrade['price']/groupByGrade['price'].sum() * 2 * pi
# groupByGrade['color'] = Category20c[len(groupByGrade.index)]
# groupByGrade['index'] = groupByGrade.index
# p1 = figure(plot_height=350, title="House average price and categories distribution",
#             toolbar_location=None, tools='hover')
# p1.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True),
#          end_angle=cumsum('angle'), line_color="black", fill_color='color',
#          legend='grade', source=groupByGrade)
# hover = p1.select(dict(type=HoverTool))
# hover.tooltips = [('Average price', '@price{int}'), ('Grade', '@grade')]
# show(p1)

# groupByCondition = housing[['price', 'condition']].groupby(['condition']).count()
# groupByCondition['index'] = groupByCondition.index
# hist, edges = np.histogram(groupByCondition, density=True, bins=len(groupByCondition.index))
# p2 = figure(y_axis_type='linear')
# p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color='white')
# show(p2)

housing['coordinates_x'] = ""
housing['coordinates_y'] = ""
for i in range(len(housing)):
        print(housing['lat'].loc[i])
        housing['coordinates_x'].loc[i] = merc(housing['lat'].loc[i], housing['long'].loc[i])[0]
        housing['coordinates_y'].loc[i] = merc(housing['lat'].loc[i], housing['long'].loc[i])[1]
p = figure(x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(CARTODBPOSITRON)
p.circle(x=housing['coordinates_x'], y=housing['coordinates_y'],  line_color="#FF0000",
         fill_color="#FF0000", fill_alpha=0.05)
show(p)






print('siema')
