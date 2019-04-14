import numpy as np
import pandas as pd
from matplotlib import *
import matplotlib.pyplot as plt
from pandas import DataFrame
import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mstats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor


# Where to save the figures
PROJECT_ROOT_DIR = r"D:\Chapter 2"
CHAPTER_ID = "projekt"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


now = datetime.datetime.now()


housing = pd.read_csv(r'house.csv')
#Constant values
valueOfSqM = 10.76
numberOfBins = 4
divider = 1000
threshold = 3
low = .05
high = .95


#<editor-fold desc='Droping unnecesary columns'>
housing = housing.drop(["sqft_living15", "sqft_lot15", 'date'], axis=1)
#</editor-fold>
#<editor-fold desc='Creating dolumns which represnet data in square meters'>
housing['sqm_living'] = round(housing['sqft_living']/valueOfSqM)
housing['sqm_lot'] = round(housing['sqft_lot']/valueOfSqM)
housing['sqm_above'] = round(housing['sqft_above']/valueOfSqM)
housing['sqm_basement'] = round(housing['sqft_basement']/valueOfSqM)
#</editor-fold>
#<editor-fold desc='Droping columns which represent data in square foots'>
housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)
#</editor-fold>

#Creating a price category
maximum = housing['price'].max()/divider
minimum = housing['price'].min()/divider

#Creating bins using the price category
def create_bins(maximum, minimum):
    x= [maximum+1]
    difference = (maximum-minimum)/numberOfBins
    for i in range(numberOfBins):
        if i == 0:
            x.append(int((minimum-1 +(difference*i))))
        else:
            x.append(int((minimum + (difference * i))))
    x.sort()
    return x
housing['price_cat'] = pd.cut(housing['price']/divider, bins= create_bins(maximum, minimum), labels = [1,2,3,4])

#Spliting data in train and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["price_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Deleting price_category from train and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)


housing = strat_train_set.copy()
#Creating copy of dataset for virtualization
virt = housing.copy()
virt['all_rooms'] = virt['bathrooms'] + virt['bedrooms']
virt.loc[virt.all_rooms== 0, 'all_rooms'] = 1
virt['avg_room_size'] = virt['sqm_living']/ virt['all_rooms']
virt['avg_floor_sq'] = virt['sqm_above'] / virt['floors']
virt['was_seen'] = virt.loc[virt.view > 0, 'was_seen'] = 1

#virt['yearsFromLastRenovation'] =  virt['yr_renovated'].apply(lambda x: int(now.year - x) if x > 0 else np.nan)
corr_matrix = virt.corr()

#Create diagrams for most promising correlation with price
#Most 4: sqm_lot, sqm_living, grade, sqm_above, avgRoomSize
#<editor-fold desc='sqm_basement_ plot'>
sns.boxplot(x=virt['sqm_basement'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['sqm_basement'], virt['price'])
ax.set_xlabel('Square meters basement')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
'''
#<editor-fold desc='long plot'>
sns.boxplot(x=virt['long'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['long'], virt['price'])
ax.set_xlabel('Long')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='lat plot'>
sns.boxplot(x=virt['lat'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['lat'], virt['price'])
ax.set_xlabel('Latitude')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='zipcode plot'>
sns.boxplot(x=virt['zipcode'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['zipcode'], virt['price'])
ax.set_xlabel('Zipcode')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='yr_renovated plot'>
sns.boxplot(x=virt['yr_renovated'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['yr_renovated'], virt['price'])
ax.set_xlabel('Year renovated')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='yr_built plot'>
sns.boxplot(x=virt['yr_built'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['yr_built'], virt['price'])
ax.set_xlabel('Year built')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='bedrooms plot'>
sns.boxplot(x=virt['bedrooms'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['bedrooms'], virt['price'])
ax.set_xlabel('No. bedrooms')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='floors plot'>
sns.boxplot(x=virt['floors'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['floors'], virt['price'])
ax.set_xlabel('No. floors')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='waterfront'>
sns.boxplot(x=virt['waterfront'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['waterfront'], virt['price'])
ax.set_xlabel('Waterfront')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='bathrooms'>
sns.boxplot(x=virt['bathrooms'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['bathrooms'], virt['price'])
ax.set_xlabel('No. bathrooms')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='condition'>
sns.boxplot(x=virt['condition'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['condition'], virt['price'])
ax.set_xlabel('Condition')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='sqm_lot plot'>
sns.boxplot(x=virt['sqm_lot'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['sqm_lot'], virt['price'])
ax.set_xlabel('Square meters of lot')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='sqm_living'>
sns.boxplot(x=virt['sqm_living'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['sqm_living'], virt['price'])
ax.set_xlabel('House square maters living')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='grade'>
sns.boxplot(x=virt['grade'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['grade'], virt['price'])
ax.set_xlabel('House grade')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='sqm_above'>
sns.boxplot(x=virt['sqm_above'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['sqm_above'], virt['price'])
ax.set_xlabel('House square maters without basement')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>
#<editor-fold desc='avgRoomSize'>
sns.boxplot(x=virt['avgRoomSize'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(virt['avgRoomSize'], virt['price'])
ax.set_xlabel('Average size of room')
ax.set_ylabel('House price')
plt.show()
#</editor-fold>

'''


virt_num = virt


#Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = virt_num.quantile(0.25)
    Q3 = virt_num.quantile(0.75)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# filtered = get_rid_of_outliers(virt_num[['price', 'sqm_basement', 'sqm_above',
#                                          'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']])
# corr_matrix_filtered = filtered.corr()
#
# '''
# #<editor-fold desc='Filtered: sqm_basement'>
# sns.boxplot(x=filtered['sqm_basement'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['sqm_basement'], filtered['price'])
# ax.set_xlabel('Square meters of basement')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
# #<editor-fold desc='Filtered: sqm_living'>
# sns.boxplot(x=filtered['sqm_living'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['sqm_living'], filtered['price'])
# ax.set_xlabel('House square maters living')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
# #<editor-fold desc='Filtered: sqm_above'>
# sns.boxplot(x=filtered['sqm_above'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['sqm_above'], filtered['price'])
# ax.set_xlabel('House square meters above.')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
# #<editor-fold desc='Filtered: sqm_lot'>
# sns.boxplot(x=filtered['sqm_lot'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['sqm_lot'], filtered['price'])
# ax.set_xlabel('House square maters lot')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
# #<editor-fold desc='Filtered: bedrooms'>
# sns.boxplot(x=filtered['bedrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['bedrooms'], filtered['price'])
# ax.set_xlabel('No. bedrooms')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
# #<editor-fold desc='Filtered: bathrooms'>
# sns.boxplot(x=filtered['bathrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(filtered['bathrooms'], filtered['price'])
# ax.set_xlabel('No. bathrooms')
# ax.set_ylabel('House price')
# plt.show()
# #</editor-fold>
#
# '''
# result = pd.merge(filtered, virt_num, how='inner', on=['price', 'sqm_basement', 'sqm_above',
#                                          'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
# result_nd = DataFrame.drop_duplicates(result)


#prepare for machine learning
#housing = strat_train_set.drop(['id'], axis=1)
housing_labels = strat_train_set["price"].copy()

#Get only numeric values
#housing_num = strat_train_set.drop("date", axis=1)
#Distinguish housing categories for processing : get year and month


# def transform_categorital(data):
#     data_cat = pd.get_dummies(data['date'].apply(lambda x: x.replace(x, x[0:6])), prefix = 'date')
#     return_data = pd.concat([data, data_cat], axis=1)
#     return return_data.drop('date',axis=1)

def add_additional_attributes(data):
    data['all_rooms'] = data['bathrooms'] + data['bedrooms']
    data.loc[data.all_rooms== 0, 'all_rooms'] = 1
    data['avg_room_size'] = data['sqm_living']/ data['all_rooms']
    data['avg_floor_sq'] = data['sqm_above'] / data['floors']
    return data

def transform_data(data):
    data_filtered = get_rid_of_outliers(add_additional_attributes(data)[['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']])
    data_connected = pd.merge(data_filtered, data, how='inner', on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_deleted_columns = data_no_duplicates.drop(['bathrooms', 'bedrooms', 'floors'], axis=1)
    cols = [col for col in data_deleted_columns.columns if col not in ['price', 'id']]
    data_scaled = StandardScaler().fit_transform(data_no_duplicates[cols])
    return data_scaled


def get_labels(data):
    data_filtered = get_rid_of_outliers(add_additional_attributes(data)[['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms','sqm_living']])
    data_connected = pd.merge(data_filtered, data, how='inner',on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms','bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_prepared = data_no_duplicates['price']
    to_return = data_prepared.values
    return to_return


housing_prepared = transform_data(housing)
housing_labels = get_labels((housing))

# #Select and train a model
#
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared.values, housing_labels.values)
#
# some_data = housing
# some_labels = housing_labels
# some_data_prepared = transform_data(transform_categorital(some_data))
#
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
#
#
#
#
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)
#
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
#
#
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
#
# display_scores(tree_rmse_scores)
#
#




forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
scoresRandomForestRegression =cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rforest_rmse_scores = np.sqrt(-scoresRandomForestRegression)
display_scores(rforest_rmse_scores)




# svm_reg = SVR(kernel="rbf")
# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# scoresSVR =cross_val_score(svm_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# svr_scores = np.sqrt(-scoresSVR)
# display_scores(svr_scores)


#
#
#
#
 param_grid = [
     # try 12 (3×4) combinations of hyperparameters
     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
     # then try 6 (2×3) combinations with bootstrap set as False
     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
   ]

 # depths = np.arange(1, 21)
 # num_leafs = [1, 5, 10, 20, 50, 100]
 # param_grid = [{'decisiontreeregressor__max_depth':depths,
 #               'decisiontreeregressor__min_samples_leaf':num_leafs}]
 # pipe_tree = make_pipeline(tree_reg)

 # gs = GridSearchCV(estimator=forest_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
#  # gs.fit(housing_prepared, housing_labels)
#  # cvres = gs.cv_results_
#  # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#  #    print(np.sqrt(-mean_score), params)

'''
# <editor-fold>
# forest_reg = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error', return_train_score=True)
#  grid_search.fit(housing_prepared, housing_labels)
#  cvres = grid_search.cv_results_
#  for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#      print(np.sqrt(-mean_score), params)
#  print("#########")
# </editor-fold>
# 
# 
# <editor-fold>
#  param_distribs = {
#          'n_estimators': randint(low=1, high=200),
#          'max_features': randint(low=1, high=8),
#      }
# 
#  forest_reg = RandomForestRegressor(random_state=42)
#  rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                  n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
#  rnd_search.fit(housing_prepared, housing_labels)
#  cvres = rnd_search.cv_results_
#  for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#      print(np.sqrt(-mean_score), params)
#  </editor-fold


'''