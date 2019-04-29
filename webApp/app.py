from flask import Flask
import pickle
import numpy as np
from flask_cors import CORS
from flask import Flask, render_template, redirect, url_for, request, jsonify
import database.house as house_db
import dbConnection as db
import create_Tokens as account
import json
from bson import json_util
from bokeh.embed import components
from bson.json_util import dumps

app = Flask(__name__)
cors = CORS(app)
app.static_folder = 'static'


def createCharts():
    import numpy as np
    from bokeh.io import output_notebook, output_file, show
    from bokeh.plotting import figure, show, gmap
    from bokeh.layouts import column
    from bokeh.models.glyphs import Quad
    from bokeh.models import ColumnDataSource, GMapOptions
    import pandas as pd
    from datetime import datetime
    from math import pi
    from bokeh.transform import cumsum
    from bokeh.models import HoverTool, BasicTickFormatter
    from bokeh.palettes import Category20c
    from bokeh.tile_providers import CARTODBPOSITRON
    import math
    from bokeh.embed import components
    from ast import literal_eval

    def merc(lat, lon):
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x / lon
        y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 +
                                                lat * (math.pi / 180.0) / 2.0)) * scale
        return (x, y)

    housing_n = db.get_data()
    valueOfSqM = 10.76
    housing_n = housing_n.drop(["sqft_living15", "sqft_lot15", 'waterfront'], axis=1)
    housing_n['floors'] = housing_n['floors'].str[1:-1]
    housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
    housing_n['date'] = housing_n['date'].apply(lambda x: x.replace(x, x[0:10]))
    housing = housing_n.convert_objects(convert_numeric=True)
    housing['sqm_living'] = round(housing['sqft_living'] / valueOfSqM)
    housing['sqm_lot'] = round(housing['sqft_lot'] / valueOfSqM)
    housing['sqm_above'] = round(housing['sqft_above'] / valueOfSqM)
    housing['sqm_basement'] = round(housing['sqft_basement'] / valueOfSqM)
    housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)
    housing['date'] = housing['date'].apply(lambda x: x.replace(x, x[1:5] + "-" + x[5:7] + "-" + x[7:9]))
    housing['date'] = housing['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

    # pierwszy
    groupByGrade = housing[['price', 'grade']].groupby(['grade']).mean()
    legend_title = "House grades"
    groupByGrade['angle'] = groupByGrade['price'] / groupByGrade['price'].sum() * 2 * pi
    groupByGrade['color'] = Category20c[len(groupByGrade.index)]
    groupByGrade['index'] = groupByGrade.index
    p1 = figure(plot_height=350, title="House average price and categories distribution",
                toolbar_location=None, tools='hover')
    p1.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True),
             end_angle=cumsum('angle'), line_color="black", fill_color='color',
             legend='grade', source=groupByGrade)
    hover = p1.select(dict(type=HoverTool))
    hover.tooltips = [('Average price', '@price{int}'), ('Grade', '@grade')]

    # drugi
    # housing['coordinates_x'] = ""
    # housing['coordinates_y'] = ""
    # for i in range(len(housing)):
    #         housing['coordinates_x'].loc[i] = merc(housing['lat'].loc[i], housing['long'].loc[i])[0]
    #         housing['coordinates_y'].loc[i] = merc(housing['lat'].loc[i], housing['long'].loc[i])[1]
    # p2 = figure(x_axis_type="mercator", y_axis_type="mercator")
    # p2.add_tile(CARTODBPOSITRON)
    # p2.circle(x=housing['coordinates_x'], y=housing['coordinates_y'],  line_color="#FF0000",
    #          fill_color="#FF0000", fill_alpha=0.05)

    # trzecie
    condition = housing[['condition']].sort_values(by='condition')
    t = pd.DataFrame(condition['condition'].value_counts())
    t['index'] = t.index
    t['index'] = t['index'].astype(str)
    t['condition'] = t['condition'].astype(str)
    p3 = figure(x_range=t['index'], plot_height=250, title='Number of houses of various conditions')
    p3.vbar(x=t['index'], top=t['condition'], width=0.9)
    p3.add_tools(HoverTool(tooltips=[("Condition", "@x"), ('Number of houses', '@y{int}')]))
    p3.xgrid.grid_line_color = None
    p3.y_range.start = 0

    # czwarte
    groupByFloors = housing[['sqm_living', 'sqm_above', 'sqm_basement', 'sqm_lot', 'floors']].groupby(['floors']).mean()
    groupByFloors['index'] = groupByFloors.index
    p4 = figure(plot_width=1000, plot_height=1000)
    p4.line(list(groupByFloors['index']), list(groupByFloors['sqm_living']), line_width=2, color="#A6CEE3",
            legend='Average square meters living')
    p4.line(list(groupByFloors['index']), list(groupByFloors['sqm_above']), line_width=2, color="#B2DF8A",
            legend='Average square meters above')
    p4.line(list(groupByFloors['index']), list(groupByFloors['sqm_basement']), line_width=2, color="#33A02C",
            legend='Average square meters basement')
    p4.line(list(groupByFloors['index']), list(groupByFloors['sqm_lot']), line_width=2, color="#FB9A99",
            legend='Average square meters lot')
    p4.add_tools(HoverTool(tooltips=[("Number of floors", "@x"), ('Value', '@y{int}')]))

    # piąte
    groupByZipcode = housing[['zipcode', 'price']].groupby(['zipcode']).mean().sort_values(by='price', ascending=False)[
                     0:10]
    groupByZipcode['zipcode'] = groupByZipcode.index
    groupByZipcode['price'] = groupByZipcode['price'].astype(str)
    groupByZipcode['zipcode'] = groupByZipcode['zipcode'].astype(str)
    p5 = figure(x_range=groupByZipcode['zipcode'], plot_height=250, title='Top 10 most richest average neighbourhoods')
    p5.vbar(x=groupByZipcode['zipcode'], top=groupByZipcode['price'], width=0.9)
    p5.add_tools(HoverTool(tooltips=[("Zipcode", "@x"), ('Average price', '@y{int}')]))
    p5.xgrid.grid_line_color = None
    p5.y_range.start = 0

    # szóste
    groupByRenovated = housing[['yr_renovated', 'price']].groupby(['yr_renovated']).count()
    groupByRenovated['index'] = groupByRenovated.index
    numberOfNotRenovated = groupByRenovated.where(groupByRenovated['index'] == 0)['price'].sum().sum()
    numberOfRenovated = groupByRenovated.where(groupByRenovated['index'] > 0)['price'].sum().sum()
    # groupByGrade.append(groupByGrade.where('index'> 0)['price'.count(), 2)
    dataframe = pd.DataFrame([['renovated', numberOfRenovated], ['not-renovated', numberOfNotRenovated]],
                             columns=['type', 'number'])

    legend_title = "Number of renovated and not renovated houses"
    w = dataframe['number'].sum()
    dataframe['angle'] = dataframe['number'] / dataframe['number'].sum() * 2 * pi

    dataframe['color'] = ['green', 'orange']
    p6 = figure(plot_height=350, title="Number of renovated and not renovated houses", toolbar_location=None,
                tools='hover')
    p6.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True),
             end_angle=cumsum('angle'), line_color="black", fill_color='color',
             legend='type', source=dataframe)
    hover = p6.select(dict(type=HoverTool))
    hover.tooltips = [('Type', '@type'), ('Number of houses', '@number{int}')]

    # siódme
    groupByDate = housing[['date', 'price']].groupby(['date']).count()
    groupByDate['index'] = groupByDate.index
    p7 = figure(plot_width=1000, plot_height=1000, x_axis_type='datetime')
    p7.line(groupByDate['index'], groupByDate['price'], line_width=2, color="#A6CEE3",
            legend='Number of sold houses')
    p7.add_tools(HoverTool(tooltips=[("Date", "@x{DateTime}"), ('Sold houses', '@y{int}')]))

    # ósme
    groupByPrice = housing[['date', 'price']].groupby(['date']).sum()
    groupByPrice['index'] = groupByPrice.index
    p8 = figure(plot_width=1000, plot_height=1000, x_axis_type='datetime')
    p8.yaxis.formatter = BasicTickFormatter(use_scientific=False)

    p8.line(groupByPrice['index'], groupByPrice['price'], line_width=2, color="#33A02C",
            legend='Earned money')
    p8.add_tools(HoverTool(tooltips=[("Date", "@x{DateTime}"), ('Sum from sold houses', '@y{int}')]))

    script,div = components(column(p1, p3, p4, p5, p6, p7, p8))

    return script,div

@app.route('/')
def hello_world():
    return render_template('main.html')\

@app.route('/main')
def main():
    return render_template('main.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/agent_view')
def agent_view():
    return render_template('agent_view.html')


@app.route('/register_agent')
def register_agent():
    return render_template('register_agent.html')


@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')


@app.route('/add_house', methods=['GET', 'POST'])
def add_house():
    form_value = request.form
    json_val = jsonify(form_value)
    house = house_db.create_house()
    house = house_db.set_properties(house, form_value)
    db.add_house(house)
    return render_template('add_house_result.html', result=form_value)


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    form_value = request.form
    mail = form_value["email"]
    password = form_value["password"]
    password2 = form_value["passwordCon"]
    token = form_value["token"]
    if(password == password2):
        if(account.change_password(email=mail, token=token, new_password= password)):
            return render_template('login.html')
    return render_template('forgot_password.html', result='error')


@app.route('/check_cred', methods=['GET', 'POST'])
def check_credentials():
    form_value = request.form
    username = form_value["username"]
    password = form_value["password"]
    result = account.log_in(username, password)
    if(result == True):
        return render_template('agent_view.html')
    return render_template('login.html', result='error')


@app.route('/register_agent_check', methods=['GET', 'POST'])
def register_agent_check():
    form_value = request.form
    mail = form_value["email"]
    password = form_value["password"]
    password2 = form_value["passwordCon"]
    token = form_value["token"]
    result =False
    if(password == password2):
        result = account.register(mail, password, token)
    if(result):
        return render_template('agent_view.html')
    return render_template('register.html', result='error')

@app.route('/predict', methods=['GET', 'POST'])
def register_agent_ch7eck():
    form_value = request.form

    x= model.predict(np.array(form_value[['bedrooms', 'bathrooms', 'condition', 'floors', 'grade', 'lat', 'long',  'yr_built','yr_renovated', 'zipcode']]))
    return render_template('register.html', result=''+x)


@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    FIELDS = {'price': True, 'bedrooms': True, 'bathrooms': True,
              'sqft_living': True, 'sqft_lot': True, 'date': True}
    projects = account.connect_to_houses().find(projection=FIELDS)
    json_projects=[]
    for project in projects:
        json_projects.append(project)
    json_projects = json.dumps(json_projects, default=json_util.default)
    return json_projects


@app.route('/stats', methods=['GET', 'POST'])
def stats():
    script, div = createCharts()
    return render_template("statistics.html", the_div=div, the_script=script)




if __name__ == '__main__':
    model = pickle.load(open('../analysis/model.pkl', 'rb'))
    app.run()


