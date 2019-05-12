from pymongo import MongoClient
from pandas import DataFrame
import pymongo
import pandas as pd


def connect():
    connection = MongoClient("mongodb://drugi:90809988Qwe@prediction-shard-00-00-jm5ad.mongodb.net:27017,prediction-shard-00-01-jm5ad.mongodb.net:27017,prediction-shard-00-02-jm5ad.mongodb.net:27017/test?ssl=true&replicaSet=Prediction-shard-0&authSource=admin&retryWrites=true")
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
    result = DataFrame(list(connect_to_houses().find({}, {'_id': 0,
                                                          'sqft_living15': 0, 'sqft_lot15': 0,
                                                          'waterfront': 0, 'view':0})))
    return result





