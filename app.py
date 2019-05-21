import pickle
import os
import atexit
from flask import Flask, render_template, request, jsonify,redirect
from database import house as house_db
from database import manage as account
from database import dbConnection as conn
import prepare_for_prediction as pred
from database import create_Tokens as tokens
from analysis.dashboard_diagrams import diagrams
from analysis import analysis_main
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from apscheduler.schedulers.background import BackgroundScheduler

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
app.static_folder = 'static'
app.config.update(
    SECRET_KEY='sdf'
)

login = LoginManager()

login.init_app(app)



@login.unauthorized_handler
def unauthorized():
    return render_template('login.html',
                           result='Login first to access this page.')


class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @property
    def is_admin(self):
        user = self.get_user()
        w = user['is_admin']
        return w

    def get_user(self):
        return account.get_user_by_id(self.id)


@login.user_loader
def load_user(user_id):
    return User(user_id)


@app.before_first_request
def init_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(analysis_main.model_preparation, 'interval', minutes=2)
    # hours = 24, start_date = '2019-05-19 15:25:00'
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


@app.route('/')
def main2():
   return render_template('main.html')



@app.route('/main')
def main():
    return render_template('main.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        form = request.form
        email = request.form['email']
        password = request.form['password']
        if account.log_in(email, password) is True:
            user = User(account.get_user_by_mail(email)['token']['_id'])
            login_user(user)
            return redirect('main')
        else:
            return render_template('login.html', result='Wrong password or email.')
    else:
        return render_template('login.html')


@app.route('/logout', methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return render_template('main.html')



@app.route('/agent_view')
@login_required
def agent_view():
    return render_template('agent_view.html')


@app.route('/add_house', methods=['GET', 'POST'])
@login_required
def add_house():
    form_value = request.form
    json_val = jsonify(form_value)
    house = house_db.create_house()
    house = house_db.set_properties(house, form_value)
    house_db.add_house(house)
    return render_template('add_house_result.html', result=form_value)


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        form_value = request.form
        mail = form_value["email"]
        password = form_value["password"]
        password2 = form_value["passwordCon"]
        token = form_value["token"]
        if password == password2:
            user = account.connect_to_users().find_one({"email": mail})
            is_the_same = account.check_password(password, user['password'])
            if not is_the_same:
                if account.change_password(email=mail, token=token, new_password=password) is True:
                    logout_user()
                    return render_template('login.html', result="Use new password to log in.")
                else:
                    return render_template('change_password.html', result='The token is incorrect.')
            else:
                return render_template('change_password.html', result="Can't change password for the same.")
        else:
            return render_template('change_password.html', result='Passwords are not the same.')
    else:
        return render_template('change_password.html')


@app.route('/reset_password', methods=['GET', 'POST'])
@login_required
def reset_password():
    if request.method == 'POST':
        form_value = request.form
        mail = form_value["email"]
        token = form_value["token"]
        if account.reset_password(token=token, email=mail) is True:
            logout_user()
            return render_template('login.html',
                                   result="Your password has been reset and sent on your e-mail. Use it to log in.")
        else:
            return render_template('reset_password.html', result='Token or e-mail are incorrect.')
    else:
        return render_template('reset_password.html')


@app.route('/send_token', methods=['GET', 'POST'])
@login_required
def send_token():
    if request.method == 'POST':
        form_value = request.form
        mail = form_value["email"]
        number = form_value["token"]
        result = tokens.give_token(mail, number)
        if result is True:
            return render_template('send_token.html', token=tokens.get_token(),
                                   message="You have successfully sent the token.")
        else:
            return render_template('send_token.html', token=tokens.get_token(),
                                   message="The email is already registered or the token is incorrect.")
    else:
        number_ = conn.connect_to_tokens().count({"isUsed": False})
        return render_template('send_token.html', token=tokens.get_token(), tokens=number_)


@app.route('/add_tokens', methods=['POST'])
@login_required
def add_tokens():
    result = tokens.insert_1000_tokens()
    if result is True:
        return redirect(send_token)
    else:
        add_tokens()


@app.route('/register_agent', methods=['GET', 'POST'])
def register_agent():
    if request.method == 'POST':
        form_value = request.form
        mail = form_value["email"]
        password = form_value["password"]
        password2 = form_value["passwordCon"]
        token = form_value["token"]
        result = False
        if password == password2:
            result = account.register(mail, password, token)
        if result:
            return render_template('login.html')
        return render_template('register_agent.html', result='There was an error with registration. Try again.')
    else:
        return render_template('register_agent.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form_value = request.form.to_dict()
    data = pred.prepare_data(form_value)
    return render_template('main.html', data=model.predict(data))


@app.route('/delete_account', methods=['GET', 'POST'])
@login_required
def delete_account():
    if request.method == 'POST':
        form_value = request.form
        mail = form_value["email"]
        password = form_value["password"]
        password2 = form_value["passwordCon"]
        token = form_value["token"]
        if password == password2:
                result_ = account.delete_account(token, password, mail)
                if result_ is True:
                    logout_user()
                    return render_template('login.html',
                                           result="Your account has been deleted")
                else:
                    return render_template('delete_account.html', result="An error while deleting you account")
        else:
            return render_template('delete_account.html', result= "Passwords are not the same")

    return render_template('delete_account.html')


@app.route('/statistics', methods=['GET', 'POST'])
@login_required
def statistics():
    script, div = diagrams.make_diagrams()
    return render_template("statistics.html", the_div=div, the_script=script)


if __name__ == '__main__':
    app.run()

