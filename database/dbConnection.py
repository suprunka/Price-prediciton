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
    found_result = False
    result = connect_to_database().find_one({'id': '"%s"' % id})
    if result is None:
        another_attempt = connect_to_database().find_one({'id': '""%s""' % id})
        if another_attempt is None:
            return found_result
    found_result = True
    return found_result


def delete_specific(id):
    delete_result = False
    result = connect_to_database().delete_one({'id': '"%s"'%id})
    if result.deleted_count == 0:
        another_attempt = connect_to_database().delete_one({'id': '""%s""'%id})
        if another_attempt.deleted_count == 0:
            return delete_result
    delete_result = True
    return delete_result



def add_house():
    connect_to_database().insert_one(set_properties(create_house(), {'lat': 343, 'price':324530, 'sqm_lot': 320}).__dict__)


add_house()

# t = delete_specific(978432)
# print(t)



