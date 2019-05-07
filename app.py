from flask import Flask
import pickle
import numpy as np
from flask_cors import CORS
from flask import Flask, render_template, redirect, url_for, request, jsonify
from database import house as house_db
from database import dbConnection as db
from database import  create_Tokens as account
import prepare_for_prediction as pred
import json
from bson import json_util
from analysis.dashboard_diagrams import *
from bokeh.embed import components
from bson.json_util import dumps
from analysis import averaged_models
from flask_login import LoginManager

with open('modelfin.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
cors = CORS(app)
app.static_folder = 'static'
login = LoginManager(app)
login.init_app(app)
app.config.update(
    SECRET_KEY='secret_key_iksde'
)


@login.user_loader
def load_user(user_id):
    return User(user_id)


class User(UserMixin):
    def __init__(self, id):
        self.id = id


def load_user(id):
    return User(account.get_user(id))


@app.route('/')
def hello_world():
    return render_template('main.html')\

@app.route('/main')
def main():
    return render_template('main.html')



@app.route('/logout', methods=["GET", "POST"])
def logout():
    logout_user()
    return render_template('main.html')


@app.route('/agent_view')
@login_required
def agent_view():
    return render_template('agent_view.html')


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
@login_required
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
    username = form_value["email"]
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
    form_value = request.form.to_dict()
    data = pred.prepare_data(form_value)
    x = model.predict(data)
    result = x[0]
    return render_template('main.html',data= result)


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
@login_required
def stats():
    script, div = make_diagrams()
    return render_template("statistics.html", the_div=div, the_script=script)

if __name__ == '__main__':
    app.run()


