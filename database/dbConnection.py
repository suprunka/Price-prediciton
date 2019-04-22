from pymongo import MongoClient
from pandas import DataFrame
import pymongo
from house import *

def connect_to_database():
    connection = MongoClient("mongodb://jakub:90809988Qwe@thecluster-shard-00-00-zrxzv.mongodb.net:27017,thecluster-shard-00-01-zrxzv.mongodb.net:27017,thecluster-shard-00-02-zrxzv.mongodb.net:27017/test?ssl=true&replicaSet=theCluster-shard-0&authSource=admin&retryWrites=true")
    db = pymongo.database.Database(connection, 'Project')
    collection = pymongo.collection.Collection(db, 'Houses')
    return collection



def get_data():
    result = DataFrame(list(connect_to_database().find({}, {'_id': 0})))
    return  result


def get_specific(id):
    result = connect_to_database().find({'id': '"%s"'%id})
    return result


def delete_specific(id):
    result = connect_to_database().delete_one({'id': '"%s"'%id})


def add_house_test():
    connect_to_database().insert_one(set_properties(create_house(), {'price': 53200, 'lat': 123}).__dict__)

def add_house(house):
    connect_to_database().insert_one(house.__dict__)


add_house_test()
