from pymongo import MongoClient
from pandas import DataFrame
import pymongo
import pandas as pd


def connect():
    connection = MongoClient("mongodb+srv://jakub23:90809988Qwe@prediction-jm5ad.mongodb.net/test?retryWrites=true")

    return pymongo.database.Database(connection, 'Project')


def connect_to_tokens():
    collection = pymongo.collection.Collection(connect(), 'Tokens')
    return collection


def connect_to_users():
    collection = pymongo.collection.Collection(connect(), 'Users')
    return collection


def connect_to_houses():
    collection = pymongo.collection.Collection(connect(), 'Houses')
    return collection


def get_data():
    result = DataFrame(list(connect_to_houses().find({}, {'_id': 0, 'id':0,
                                                          'sqft_living15': 0, 'sqft_lot15': 0,
                                                          'waterfront': 0, 'view':0})))
    return result





