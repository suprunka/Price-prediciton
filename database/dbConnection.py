from pymongo import MongoClient

import pprint


def connect_to_database():
    db = MongoClient("mongodb+srv://jakub:90809988Qwe@thecluster-zrxzv.mongodb.net/test?retryWrites=true").Houses
    collection = db.Train
    return collection


def find_one(query):
    return connect_to_database().find(query)


def main():
    query = {"MSSubClass": "60"}
    found = find_one(query)
    pprint.pprint(found)


if __name__ == "__main__": main()


