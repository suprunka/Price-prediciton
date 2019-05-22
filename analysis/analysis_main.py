import numpy as np
import pandas as pd
import pickle
import datetime
import math
import lightgbm as lgb
import xgboost as xgb
from analysis_helpers.averaged_models import StackingAveragedModels
from analysis_helpers.models_gridSearch import gridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from analysis.analysis_helpers import averaged_models, models_gridSearch
from database.dbConnection import get_data

valueOfSqM = 10.76
numberOfBins = 4
divider = 1000

def model_preparation():
    housing_n = get_data()
    housing_n['floors'] = housing_n['floors'].str[1:-1]
    housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
    housing = housing_n.convert_objects(convert_numeric=True)
    housing_out = housing[['price', 'sqft_lot', 'sqft_living', 'bathrooms']]
    housing_filtered = get_rid_of_outliers(housing_out)
    data_connected = pd.merge(housing_filtered, housing, how='inner',

                              on=['price', 'sqft_lot', 'sqft_living', 'bathrooms'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    housing = convert_to_sqm(data_no_duplicates)
    housing = housing.rename(index=str, columns={'price_x': 'price'})  #
    housing = housing.reset_index(drop=True)
    maximum = housing['price'].max() / divider
    minimum = housing['price'].min() / divider
    housing['price_cat'] = pd.cut(housing['price'] / divider, bins=create_bins(maximum, minimum), labels=[1, 2, 3, 4])

    split = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["price_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Deleting price_category from train and test sets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("price_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing_prepared = transform_data(housing)
    housing_labels = get_labels(housing)

    # Picking most promising models
    forest_reg = RandomForestRegressor(n_jobs=-1, min_weight_fraction_leaf=0., n_estimators=100, min_samples_split=16,
                                       min_samples_leaf=8, min_impurity_decrease=0, max_depth=100)

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    model_xgb = xgb.XGBRegressor()

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    knn = KNeighborsRegressor(n_neighbors=8)
    model_xgb.fit(housing_prepared, housing_labels)
    pickle.dump(model_xgb, open('./xgb_model.pkl', 'wb'))
    print('Prediction model was updated.')


def convert_to_sqm(housing):
    housing['sqm_living'] = round(housing['sqft_living']/valueOfSqM)
    housing['sqm_lot'] = round(housing['sqft_lot']/valueOfSqM)
    housing['sqm_above'] = round(housing['sqft_above']/valueOfSqM)
    housing['sqm_basement'] = round(housing['sqft_basement']/valueOfSqM)
    housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)
    return housing


def create_bins(maximum, minimum):
    x = [maximum+1]
    difference = (maximum-minimum)/numberOfBins
    for i in range(numberOfBins):
        if i == 0:
            x.append(int((minimum-1 + (difference * i))))
        else:
            x.append(int((minimum + (difference * i))))
    x.sort()
    return x


def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.1)
    Q3 = num_data.quantile(0.9)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR))).any(axis=1)]


def add_additional_attributes(data):
    data['floors'] = data['floors'].astype(float)
    data['all_rooms'] = data['bathrooms'] + data['bedrooms']
    data.loc[data.all_rooms == 0, 'all_rooms'] = 1
    data['avg_room_size'] = data['sqm_living']/ data['all_rooms']
    data['avg_floor_sq'] = data['sqm_above'] / data['floors']
    data['overall'] = data['grade'] + data['condition']
    x = data[["zipcode", "price"]].groupby(['zipcode'], as_index=False).mean().sort_values(by='price', ascending=False)
    x['zipcode_cat'] = 0
    x['zipcode_cat'] = np.where(x['price'] > 1000000, 3, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(750000, 1000000), 2, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(500000, 750000), 1, x['zipcode_cat'])
    x = x.drop('price', axis=1)
    data = pd.merge(data, x, how='inner',on='zipcode')
    data = data.drop('zipcode', axis=1)
    train_set = data.copy()
    train_set['age'] = datetime.date.today().year - train_set['yr_built']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train_set['binned_age'] = pd.cut(train_set['age'], bins=bins, labels=labels)
    y = train_set[["binned_age", "price"]].groupby(["binned_age"], as_index=False).mean().sort_values(by='price',
                                                                                                      ascending=False)
    data['binned_age'] = 0
    data['binned_age'] = np.where(train_set['binned_age'].between(4, 9), 1, data['binned_age'])
    return data


def transform_data(data):
    data_additional_attributes = add_additional_attributes(data)
    data_deleted_columns = data_additional_attributes.drop(['id', 'price','binned_age',  'date', 'bedrooms', 'condition'], axis=1)
    cols = [col for col in data_deleted_columns.columns if col not in ['price', 'id']]
    data_deleted_columns = data_deleted_columns[['sqm_basement', 'sqm_above', 'sqm_lot', 'sqm_living', 'grade',
                                                 'yr_built', 'lat', 'long', 'all_rooms', 'avg_room_size', 'avg_floor_sq',
                                                 'overall','floors', 'zipcode_cat', 'yr_renovated',
                                                 'bathrooms']].reset_index(drop=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_deleted_columns)
    with open("./scaler.pkl", "wb") as outfile:
        pickle.dump(scaler, outfile)
    print('Scaler was updated.')
    return data_scaled


def get_labels(data):
    data_additional_attributes = add_additional_attributes(data)
    data_prepared = data_additional_attributes['price']
    to_return = data_prepared
    return to_return


def display_scores(scores, model):
    print(model)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('')


def checkAllModels(models_list,housing_prepared,housing_labels,strat_test_set,  alone=False):
    w = strat_test_set.copy()
    test_X = transform_data(w)
    test_y = get_labels(w)
    if not alone:
        for model in models_list:
            model.fit(housing_prepared, housing_labels)
            display_scores(cross_val_score(model, test_X, test_y),  model.__class__.__name__)

    else:
        models_list.fit(housing_prepared, housing_labels)
        display_scores(cross_val_score(models_list, test_X, test_y), models_list.__class__.__name__)



    #Check models
    # checkAllModels(model_xgb, alone=True)


    #Calculate accurancy
    # calculate_accurancy()

    #Grid search
    # models_gridSearch.gridSearchCV(housing_prepared, housing_labels)


    # with open("scaler.pkl", "rb") as infile:
    #     scaler = pickle.load(infile)
    #     scaled1 = scaler.transform(list1)
    #     scaled2 = scaler.transform(list2)
    #     # print(cross_val_score(model_xgb, scaled, test_y).mean() * 100)
    #     with open("xgb_model.pkl", "rb") as model__:
    #          model = pickle.load(model__)
    #          print(model.predict(scaled1))
    #          print(model.predict(scaled2))