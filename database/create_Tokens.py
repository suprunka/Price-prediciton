from dbConnection import *
import smtplib

# connection = connect_to_tokens()

def insert_1000_tokens():
    list = []
    for i in range(1000):
        number = round(random.randint(1, 600000) * datetime.datetime.now().microsecond+0.121/55 / 0.02)
        if number not in list:
            list.append(number)
        else:
            break
        connect_to_tokens().insert_many(list)

def get_data():
    listt = []
    token_list = list(connect_to_tokens().find({}, {'_id': 0, 'isUsed': 0}))
    for el in token_list:
        if el not in listt:
            listt.append(el['token'])

    print(len(listt))


def get_token():
    token = connect_to_tokens().find_one({'isUsed': False})
    number = token['token']
    return number

def give_token(email):
    server = smtplib.SMTP('smtp.gmail.com', 25)
    server.connect('smtp.gmail.com', 465)
    server.login('predproject2004@gmail.com','90809988Qwe')
    number = get_token()
    msg= "Hello, here is your token, use it to log in with your password: ", number
    the_token = connect_to_tokens().find_one({'token': number})
    connect_to_tokens().update(the_token, {"$set":{'isUsed':True}})
    server.send_message('predproject2004@gmail.com', email, msg)



#tworzenie tokenów do list
#wrzucenie unikowych tokenów do bazy danych
#podczas dodawania kolejnych tokenów branie tych z bazy danych
#i tworzenie nowych unikowych


give_token('jakub23sa@wp.pl')
