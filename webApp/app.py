from flask import Flask
from flask_cors import CORS
from flask import Flask, render_template, redirect, url_for, request, jsonify
import database.house as house_db
import dbConnection as db
import create_Tokens as account

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
    mail = form_value["mail"]
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
    mail = form_value["mail"]
    password = form_value["password"]
    password2 = form_value["passwordCon"]
    token = form_value["token"]
    result =False
    if(password == password2):
        result = account.register(mail, password, token)
    if(result):
        return render_template('agent_view.html')
    return render_template('register.html', result='error')



if __name__ == '__main__':
    app.run()