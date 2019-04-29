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
from analysis.dashboard_diagrams import *
from bokeh.embed import components
from bson.json_util import dumps

app = Flask(__name__)
cors = CORS(app)
app.static_folder = 'static'




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
def stats(which='p1'):
    script, div = make_diagrams(which)
    return render_template("statistics.html", the_div=div, the_script=script)


if __name__ == '__main__':
    model = pickle.load(open('../analysis/model.pkl', 'rb'))
    app.run()


