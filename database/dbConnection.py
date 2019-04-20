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


def add_house(dict):
    house = set_properties(create_house(), transform_dictionary(dict)).__dict__
    connect_to_database().insert_one(house)


def transform_dictionary(dict):
    value_one = dict.get('date')
    dict['date'] = "\"" + value_one + "\""

    value_one = dict.get('floors')
    dict['floors'] = "\"" + value_one + "\""

    value_one = dict.get('zipcode')
    dict['zipcode'] = "\"" + value_one + "\""

    return dict


dictionary = {'date': '2323123', 'price':'32042', 'bedrooms': '3', 'bathrooms': '2',
              'sqft_living': '32534', 'sqft_lot':'3212', 'floors': '2', 'waterfront': '2',
              'view': '1', 'condition': '3', 'grade': '8', 'sqft_above': '21', 'sqft_basement': '42',
              'yr_built': '321', 'yr_renovated': '213', 'zipcode': '21332', 'lat': '231', 'long': '-123',
              'sqft_living15': '321', 'sqft_lot15': '32123'}


add_house(dictionary)




# result = transform_dictionary(dictionary)
# print(result)
