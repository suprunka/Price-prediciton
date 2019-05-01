import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import dbConnection as db


def get_data_fromdb():
    return db.get_data()


def prepare_data(sent_data):
    valueOfSqM = 10.76

    sent_data['sqm_living'] = round(float(sent_data['sqft_living']) / valueOfSqM)
    sent_data['sqm_lot'] = round(float(sent_data['sqft_lot']) / valueOfSqM)
    sent_data['sqm_above'] = round(float(sent_data['sqft_above']) / valueOfSqM)
    sent_data['sqm_basement'] = round(float(sent_data['sqft_basement']) / valueOfSqM)

    data = pd.DataFrame([sent_data])
    data_additional_attributes = add_additional_attributes(data)
    data_filtered = get_rid_of_outliers(data_additional_attributes)[
        ['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner',
                              on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_deleted_columns = data_connected.drop(['bathrooms', 'yr_renovated',  'bedrooms', 'floors','condition',
                                                'sqft_above', 'sqft_lot', 'sqft_living', 'sqft_basement'], axis=1)
    cols = [col for col in data_deleted_columns.columns ]
    data_scaled = MinMaxScaler().fit_transform(data_deleted_columns[cols])
    return data_scaled


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

def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.3)
    Q3 = num_data.quantile(0.7)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]
