from dbConnection import *
import hashlib, binascii, os
import smtplib

# connection = connect_to_tokens()


def insert_1000_tokens():
    the_list = []
    for i in range(1000):
        number = round(random.randint(1, 600000) * datetime.datetime.now().microsecond+0.121/55 / 0.02)
        if number not in the_list:
            the_list.append(number)
        else:
            break
        connect_to_tokens().insert_many(the_list)


def get_data():
    the_list = []
    token_list = list(connect_to_tokens().find({}, {'_id': 0, 'isUsed': 0}))
    for el in token_list:
        if el not in the_list:
            the_list.append(el['token'])

    print(len(the_list))


def get_token():
    token = connect_to_tokens().find_one({'isUsed': False})
    number = token['token']
    return number


def give_token(email):
    number = get_token()
    if number is not None:
        the_token = connect_to_tokens().find_one({'token': number})
        msg = "dw"
        send_mail(email, msg)
        connect_to_tokens().update(the_token, {"$set": {'isUsed': True}})

    return number


def register(email, password, token):
    the_result = True
    the_token = connect_to_tokens().find_one({'token': token})
    if the_token is not None:
        result = connect_to_users().insert({'email': email,
                                            'token': the_token,
                                            'password': hash_password(password)})
        if result is None:
            the_result = False

    return the_result


def log_in(email, password):
    stored = connect_to_users().find_one({'email': email})['password']
    print(check_password(password, stored))


def change_password(email, token, new_password):
    connection = connect_to_users()
    result = connection.find_one({'token': token, 'email': email})
    if result is not None:
        connection.update_one({'token': token, 'email': email}, {'$set': {'password': hash_password(new_password)}})


def send_mail(to, message):
    server = smtplib.SMTP('smtp.gmail.com', 25)
    server.starttls()
    server.login('predproject2004@gmail.com', '90809988Qwe')
    server.sendmail('predproject2004@gmail.com', to, message)


def hash_password(password):
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt+pwdhash).decode('ascii')

def check_password(password, stored):
    salt = stored[:64]
    stored_pass = stored[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'),
                                  salt.encode('ascii'), 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_pass



# change_password('jak', 108758456820, 'mynewpassword')
# log_in('jak', 'mynewpassword')
