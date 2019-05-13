import numpy as np
import pandas as pd
import pickle
import datetime
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
import xgboost as xgb
import math
import lightgbm as lgb
import warnings
from analysis_helpers import averaged_models, models_gridSearch
warnings.filterwarnings('ignore')
from database.dbConnection import get_data



housing_n =get_data()
valueOfSqM = 10.76
numberOfBins = 4
divider = 1000


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
    data['overall'] = (data['grade'] + data['condition'])/data['yr_built']
    # data['lot-and-house'] = data['sqm_lot'] + data['sqm_living']/ data['grade']
    # data['ko'] = data['sqm_lot']/data['sqm_living']

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
    with open("scaler.pkl", "wb") as outfile:
        pickle.dump(scaler, outfile)
    return data_scaled



def transform_data_for_average_calculation(data):
    additional_attributes = add_additional_attributes(data)
    housing_out = additional_attributes[['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]

    data_filtered = get_rid_of_outliers(housing_out)
    data_connected = pd.merge(data_filtered, additional_attributes, how='inner',
                              on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])

    data_deleted_columns=data_no_duplicates[['sqm_basement', 'sqm_above', 'sqm_lot','sqm_living', 'grade', 'yr_built',
                                             'lat', 'long','all_rooms', 'avg_room_size', 'avg_floor_sq', 'overall',
                                             'zipcode_cat', 'binned_age', 'bathrooms', 'bedrooms',
                                             'condition',  'price']]

    data_deleted_columns_scale=additional_attributes[['sqm_basement', 'sqm_above', 'sqm_lot',
                                                   'sqm_living', 'grade', 'yr_built', 'lat', 'long',
                                                   'all_rooms', 'avg_room_size', 'avg_floor_sq', 'overall',
                                                   'zipcode_cat', 'binned_age','bathrooms', 'bedrooms', 'condition',
                                                   ]]

    # scaler = Normalizer()
    # scaler.fit_transform(data_deleted_columns_scale)
    # with open("scalerC.pkl", "wb") as outfile:
    #     pickle.dump(scaler, outfile)
    return data_deleted_columns



def get_labels(data):
    data_additional_attributes = add_additional_attributes(data)
    data_prepared = data_additional_attributes['price']
    to_return = data_prepared
    return to_return


housing_n['floors'] = housing_n['floors'].str[1:-1]
housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
housing = housing_n.convert_objects(convert_numeric=True)
housing_out = housing[['price','sqft_lot', 'sqft_living','bathrooms']]
housing_filtered = get_rid_of_outliers(housing_out)
data_connected = pd.merge(housing_filtered, housing, how='inner',
                          on=['price','sqft_lot', 'sqft_living', 'bathrooms'])
data_no_duplicates = data_connected.drop_duplicates(['id'])
housing = convert_to_sqm(data_no_duplicates)
housing = housing.rename(index=str, columns={'price_x': 'price'}) #
housing = housing.reset_index(drop=True)
maximum = housing['price'].max()/divider
minimum = housing['price'].min()/divider
housing['price_cat'] = pd.cut(housing['price']/divider, bins=create_bins(maximum, minimum), labels=[1, 2, 3, 4])


split = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["price_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Deleting price_category from train and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
# w = strat_test_set.copy()
housing_prepared = transform_data(housing)
housing_labels = get_labels(housing)
# test_X = transform_data(w)
# test_y = get_labels(w)


#Picking most promising models
forest_reg = RandomForestRegressor(n_jobs= -1, min_weight_fraction_leaf=0., n_estimators=100, min_samples_split=16,
                                   min_samples_leaf=8, min_impurity_decrease=0, max_depth=100)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor()


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

br_reg = BaggingRegressor(warm_start=False, n_jobs=5, n_estimators=50, bootstrap_features=True,
                          bootstrap=False)


def display_scores(scores, model):
    print(model)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('')


def checkAllModels(models_list, alone=False):
    scores_for_plot_test = []
    if not alone:
        for model in models_list:
            model.fit(housing_prepared, housing_labels)
            print("R2")
            display_scores(r2_score(test_y, model.predict(test_X)),  model.__class__.__name__)
            print("Mean absolute error")
            display_scores(mean_absolute_error(test_y, model.predict(test_X)), model.__class__.__name__)
    else:
        models_list.fit(housing_prepared, housing_labels)
        print("R2")
        display_scores(r2_score(test_y, models_list.predict(test_X)), models_list.__class__.__name__)
        print("Mean absolute error")
        display_scores(mean_absolute_error(test_y, models_list.predict(test_X)), models_list.__class__.__name__)



# models = [forest_reg, GBoost, model_xgb,  model_lgb, br_reg]
# averaged_models = averaged_models.StackingAveragedModels(base_models=(model_xgb, br_reg),
#                                                          meta_model =forest_reg)
# averaged_models.fit(housing_prepared, housing_labels)
# forest_reg.fit(housing_prepared, housing_labels)
# GBoost.fit(housing_prepared,housing_labels)
# model_xgb.fit(housing_prepared, housing_labels)
# model_lgb.fit(housing_prepared, housing_labels)
# br_reg.fit(housing_prepared, housing_labels)



list1= [[0, 299, 983, 299, 8, 2006, 47.7268, -121.95700, 7.5, 39.86667, 149.5, 0.00548, 2, 0, 0, 2.5 ]]
list2= [[0, 94, 728, 94, 6, 1977, 47.7422, -121.98100, 4, 23.5, 94, 0.00506, 1, 0, 0, 1 ]]
# with open("scaler.pkl", "rb") as infile:
#     scaler = pickle.load(infile)
#     scaled = scaler.transform(list)
#       pickle.dump(averaged_models, open('average_model.pkl', 'wb'))
# pickle.dump(model_xgb, open('xgb_model.pkl', 'wb'))
#     # pickle.dumps(forest_reg, open('forest_model.pkl', 'wb'))
#     a = averaged_models.predict(scaled)
#     # f = forest_reg.predict(scaled)
#     # gboost = GBoost.predict(scaled)
#     # xgb = model_xgb.predict(scaled)
#     # lgb = model_lgb.predict(scaled)
#     # br = br_reg.predict(scaled)
#     print(a)
#     # print(f)
#     # print(gboost)
#     # print(xgb)
#     # print(lgb)
#     # print(br)

def calculate_accurancy():
    houses = strat_test_set.copy()
    houses = houses.convert_objects(convert_numeric=True)
    houses = transform_data_for_average_calculation(houses)
    houses = houses.reset_index(drop=True)
    totaldiff = 0
    totalprice = 0
    for i in range(len(houses)):
        price = houses.loc[i,'price']
        with open("scalerC.pkl", "rb") as infile:
            scaler = pickle.load(infile)
            h = [houses.loc[i, 'sqm_basement'], houses.loc[i, 'sqm_above'], houses.loc[i, 'sqm_lot'],
                 houses.loc[i, 'sqm_living'], houses.loc[i, 'grade'], houses.loc[i, 'yr_built'], houses.loc[i, 'lat'],
                 houses.loc[i, 'long'], houses.loc[i, 'all_rooms'], houses.loc[i, 'avg_room_size'],
                 houses.loc[i, 'avg_floor_sq'], houses.loc[i, 'overall'], houses.loc[i, 'zipcode_cat'],
                 houses.loc[i, 'binned_age'], houses.loc[i, 'bathrooms'], houses.loc[i, 'bedrooms'],
                 houses.loc[i, 'condition']
            ]
            z = np.asarray(h).reshape(1, -1)
            scaled = scaler.transform(z)
            caclulated_price = model_xgb.predict(scaled)[0]
            # caclulated_price = averaged_models.predict(scaled)[0]
            # caclulated_price= forest_reg.predict(scaled)[0]
            diff = math.fabs(price - caclulated_price)
            totaldiff += diff
            totalprice += price
    x = totaldiff/totalprice
    print(1-x)




#Check models
# checkAllModels(model_xgb, alone=True)


#Calculate accurancy
# calculate_accurancy()

#Grid search
# models_gridSearch.gridSearchCV(housing_prepared, housing_labels)


with open("scaler.pkl", "rb") as infile:
    scaler = pickle.load(infile)
    scaled1 = scaler.transform(list1)
    scaled2 = scaler.transform(list2)
    # print(cross_val_score(model_xgb, scaled, test_y).mean() * 100)
    with open("xgb_model.pkl", "rb") as model__:
         model = pickle.load(model__)
         print(model.predict(scaled1))
         print(model.predict(scaled2))

