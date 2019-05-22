from database.dbConnection import *
import hashlib, binascii, os, smtplib
from bson.objectid import ObjectId
import string
import random


def register(email, password, token):
    the_result = True
    the_token = connect_to_tokens().find_one({'token': int(token)})
    if the_token is not None:
        result = connect_to_users().insert_one({'email': email, 'token': the_token,
                                                'password': hash_password(password),
                                                'is_admin': False})
        if result is None:
            the_result = False
        else:
            msg = 'Your account has been created.'
            send_mail(email, msg)
    return the_result


def get_user_by_mail(email):
    result = connect_to_users().find_one({'email': email})
    return result


def get_user_by_id(_id):
    found = connect_to_users().find_one({'token._id': ObjectId(_id)})
    return found


def change_password(email, token, new_password):
    result_ = False
    connection = connect_to_users()
    result = connection.find_one({'token.token': int(token), 'email': email})
    if result is not None:
        changed = connection.update_one({'token.token': int(token), 'email': email},
                                        {'$set': {'password': hash_password(new_password)}})
        if changed.modified_count == 1:
            msg = 'Your password has been changed to ' + new_password
            send_mail(email, msg)
            result_ = True

    return result_


def send_mail(to, message):
    server = smtplib.SMTP('smtp.gmail.com', 25)
    server.starttls()
    server.login('predproject2004@gmail.com', '90809988Qwe')
    subject = 'Housing Prediction'
    msg = "Dear " + to + ". " + message
    message = "Subject: {} \n\n {}".format(subject, msg)
    server.sendmail('predproject2004@gmail.com', to, message)
    server.quit()


def generate_random_password():
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(20))


def reset_password(token, email):
    result_ = False
    connection = connect_to_users()
    result = connection.find_one({'token.token': int(token), 'email': email})
    password = generate_random_password()
    if result is not None:
        changed = connection.update_one({'token.token': int(token), 'email': email},
                                        {'$set': {'password': hash_password(password)}})
        if changed.modified_count == 1:
            msg = 'Your password has been reset and now it is: ' + password
            send_mail(email, msg)
            result_ = True
    return result_


def hash_password(password):
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')


def log_in(email, password):
    try:
        stored = connect_to_users().find_one({'email': email})['password']
        return check_password(password, stored)
    except TypeError:
        return False


def check_password(password, stored):
    salt = stored[:64]
    stored_pass = stored[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'),
                                  salt.encode('ascii'), 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_pass


def delete_account(token, password, email):
    if log_in(email, password) is True:
        result = connect_to_users().delete_one({'token.token': int(token), 'email': email})
        if result.deleted_count == 1:
            connect_to_tokens().update_one({'token': int(token)}, {"$set": {'isUsed': False}})
            msg = 'Your account has been deleted'
            send_mail(email, msg)
            return True
        else:
            return False
    else:
        return False
