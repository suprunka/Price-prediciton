import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
from database import dbConnection as db
import pickle


def get_data_fromdb():
    return db.get_data()


def prepare_data(sent_data):
    data = pd.DataFrame(sent_data, index=[0])
    data_additional_attributes = add_additional_attributes(data)
    x = data_additional_attributes.astype(float)
    x = x[['sqm_basement', 'sqm_above', 'sqm_lot', 'sqm_living', 'grade', 'yr_built', 'lat', 'long', 'all_rooms',
           'avg_room_size', 'avg_floor_sq', 'overall','floors', 'zipcode_cat', 'yr_renovated', 'bathrooms']]
    with open("scaler.pkl", "rb") as infile:
        scaler = pickle.load(infile)
        scaled = scaler.transform(x)
    return scaled


def add_additional_attributes(data):
    data['all_rooms'] = data['bathrooms'] + data['bedrooms']
    data.loc[data.all_rooms == 0, 'all_rooms'] = 1
    data['avg_room_size'] = int(data['sqm_living'])/ int(data['all_rooms'])
    data.loc[data.floors == 0 , 'floors']=1
    data['avg_floor_sq'] = int(data['sqm_above']) / int(data['floors'])
    data['overall'] = int(data['grade']) + int(data['condition'])
    data['zipcode_cat'] =0
    zipcode3 = [98039,	98004,	98040,	98112]
    zipcode2 = [98102,98006,98109,98105,98119,98005,98075,98199]
    zipcode1 =[98033,98053,98074,98077,98177,98116,98007,98122,
                98052,98115,98027,98144,98008,98103,98029,98072,98117,
                98136,98107,98065,98024,98034]
    data.loc[data[data['zipcode'].isin(zipcode3)], 'zipcode_cat'] = 3
    data.loc[data[data['zipcode'].isin(zipcode2)], 'zipcode_cat'] = 2
    data.loc[data[data['zipcode'].isin(zipcode1)], 'zipcode_cat'] = 1
    data = data.drop('zipcode', axis=1)

    train_set = data.copy()
    train_set['age'] = datetime.date.today().year - int(train_set['yr_built'])
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train_set['binned_age'] = pd.cut(train_set['age'], bins=bins, labels=labels)

    data['binned_age'] = 0
    data['binned_age'] = np.where(train_set['binned_age'].between(4, 9), 1, data['binned_age'])
    return data

