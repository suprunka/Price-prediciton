from dbConnection import connect_to_tokens, connect_to_users
from threading import Lock
import datetime
from manage import send_mail
import random
lock = Lock()


def insert_1000_tokens():
    _result = False
    the_list = []
    existing_list = get_existing_tokens()
    for i in range(1000):
        number = round(random.randint(1, 600000) * datetime.datetime.now().microsecond+0.121/55 / 0.02)
        if not any(d['token'] == number for d in the_list) and number not in existing_list:
            the_list.append({'token': number, 'isUsed': False})
            _result = True
        else:
            #then retry to add tokens
            _result= False
            break
    if _result is True:
        connect_to_tokens().insert_many(the_list)
    return _result



def get_existing_tokens():
    the_list = []
    token_list = list(connect_to_tokens().find({}, {'_id': 0, 'isUsed': 0}))
    for el in token_list:
        if el not in the_list:
            the_list.append(el['token'])
    return the_list


def get_token():
    token = connect_to_tokens().find_one({'isUsed': False})
    number = token['token']
    return number


def give_token(email, number):
    result_ = False
    user = connect_to_users().find_one({'email':email})
    if number is not None and user is None:
        lock.acquire()
        the_token = connect_to_tokens().find_one({'token': int(number)})
        if the_token['isUsed'] is True:
            return False
        msg = "This is your token. It is important! Remember to save it. Use it to create the account: " + str(number)
        changed = connect_to_tokens().update_one(the_token, {"$set": {'isUsed': True}})
        lock.release()
        if changed.modified_count > 0:
            result_ = True
            send_mail(email, msg)
    return result_


