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
from scipy.stats import mstats, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import neighbors
from scipy import stats
from scipy.special import boxcox1p
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
import math


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
housing = housing.drop(["sqft_living15", "sqft_lot15", 'date', 'waterfront'], axis=1)
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

corr_matrix = virt.corr()


plt.hist(virt['sqm_living'], bins = 10)
plt.show()

plt.clf()
transform = np.asanyarray(virt[['sqm_living']].values)
dft = stats.boxcox(transform)[0]
plt.hist(dft, bins=10, color='red')
plt.show()



virt_num = virt


#Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.25)
    Q3 = num_data.quantile(0.75)
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
    data['was_seen'] = data.loc[data.view > 0, 'was_seen'] = 1
    x = data[["zipcode", "price"]].groupby(['zipcode'], as_index=False).mean().sort_values(by='price', ascending=False)
    x['zipcode_cat'] = 0
    x['zipcode_cat'] = np.where(x['price'] > 1000000, 3, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(750000, 1000000), 2, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(500000, 750000), 1, x['zipcode_cat'])
    x=x.drop('price', axis=1)
    data = pd.merge(x, data, how='right', on=['zipcode'])
    data=data.drop('zipcode', axis =1)

    data['avg_room_size'] = stats.boxcox(np.asanyarray(data[['avg_room_size']].values))[0]
    data['sqm_above'] = stats.boxcox(np.asanyarray(data[['sqm_above']].values))[0]
    data['sqm_lot'] = stats.boxcox(np.asanyarray(data[['sqm_lot']].values))[0]
    data['sqm_living'] = stats.boxcox(np.asanyarray(data[['sqm_living']].values))[0]


    return data

def transform_data(data):
    data_additional_attributes = add_additional_attributes(data)
    data_filtered = get_rid_of_outliers(data_additional_attributes)[
        ['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner',
                              on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_deleted_columns = data_no_duplicates.drop(['bathrooms',  'view', 'bedrooms', 'floors'], axis=1)
    cols = [col for col in data_deleted_columns.columns if col not in ['price', 'id']]
    data_scaled = MinMaxScaler().fit_transform(data_no_duplicates[cols])
    return data_scaled


def get_labels(data):
    data_additional_attributes = add_additional_attributes(data)
    data_filtered = get_rid_of_outliers(data_additional_attributes)[
        ['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner',
                              on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_prepared = data_no_duplicates['price']
    to_return = data_prepared.values
    return to_return

housing_prepared = transform_data(housing)
housing_labels = get_labels((housing))
test_X= transform_data(strat_test_set)
test_y = get_labels(strat_test_set)


# #Select and train a model
#
def display_scores(scores, model):
    print(model)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('')


def randmizedSearchCV():
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


def gridSearchCV_KNN():
    knn = neighbors.KNeighborsClassifier()

    parameters = {'n_neighbors': [4, 5, 6, 7], 'leaf_size': [1, 3, 5],
                  'algorithm': ['auto', 'kd_tree'], 'n_jobs': [-1]}

    model = GridSearchCV(knn, param_grid=parameters)
    grid_result = model.fit(housing_prepared, housing_labels)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)
scores_for_plot_train=[]
scores_for_plot_test=[]
def checkAllModels(models_list):
    for model in models_list:
        model.fit(housing_prepared, housing_labels)
        housing_predictions_train = model.predict(housing_prepared)
        model.fit(test_X, test_y)
        housing_predictions_test = model.predict(test_X)
        scores_train = np.sqrt(-cross_val_score(model, housing_prepared, housing_labels,
                                                scoring="neg_mean_squared_error", cv=10))
        scores_test = np.sqrt(-cross_val_score(model, test_X, test_y,
                                                scoring="neg_mean_squared_error", cv=10))
        scores_for_plot_train.append(scores_train.mean())
        scores_for_plot_test.append(scores_test.mean())
        display_scores(scores_train, 'Model:'+ model.__class__.__name__)
        display_scores(scores_test, 'test')
#LinearRegression
lin_reg = LinearRegression()
bayesian_ridge = linear_model.BayesianRidge()
model_ridge=linear_model.Ridge()
tree_reg = DecisionTreeRegressor(random_state=42)
forest_reg =  RandomForestRegressor(n_estimators=50, random_state=42, max_features="auto", max_depth=100,
                                   min_samples_leaf=4, bootstrap=True, min_impurity_decrease=100)
knn=neighbors.KNeighborsRegressor(5,weights='uniform')
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)




class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

def rmsle_cv_(model):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = np.sqrt(- cross_val_score(model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=cv))
    return(score)


models = [lin_reg, bayesian_ridge, model_ridge, tree_reg, forest_reg , knn, lasso,ENet, KRR,GBoost,model_xgb,  model_lgb  ]

for i in range(len(models)) :
    print(abs(len(models)-i))
    averaged_models = StackingAveragedModels(base_models=(models[abs(len(models)-10-i)],models[abs(len(models)-1-i)],models[abs(len(models)-2-i)],models[abs(len(models)-3-i)],models[abs(len(models)-4-i)],models[abs(len(models)-5-i)],models[abs(len(models)-6-i)],models[abs(len(models)-7-i)],models[abs(len(models)-8-i)],models[abs(len(models)-9-i)],models[abs(len(models)-11-i)] ), meta_model=models[abs(len(models)-12-i)])
    score = rmsle_cv_(averaged_models)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

checkAllModels(models)


#
#
# lin_reg.fit(housing_prepared, housing_labels)
# housing_predictions = lin_reg.predict(housing_prepared)
#
#
# #BayesianRidge
# bayesian_ridge = linear_model.BayesianRidge()
# bayesian_ridge.fit(housing_prepared,housing_labels)
# housing_predictions=bayesian_ridge.predict(housing_prepared)
# bayesian_ridge_scores =cross_val_score(bayesian_ridge, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# br_scores = np.sqrt(-bayesian_ridge_scores)
# display_scores(br_scores, 'BayesianRidge')
#
# #Ridge
# model_ridge=linear_model.Ridge()
# model_ridge.fit(housing_prepared,housing_labels)
# housing_predictions=model_ridge.predict(housing_prepared)
# scores = cross_val_score(model_ridge, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# model_ridge_rmse = np.sqrt(-scores)
# display_scores(model_ridge_rmse, 'ridge model')
#
# #DecisionTreeRegressor
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores, 'DecisionTreeRegressor')
#
# #RandomForestRegressor
# forest_reg =  RandomForestRegressor(n_estimators=50, random_state=42, max_features="auto", max_depth=100,
#                                    min_samples_leaf=4, bootstrap=True, min_impurity_decrease=100)
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# scoresRandomForestRegression =cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rforest_rmse_scores = np.sqrt(-scoresRandomForestRegression)
# display_scores(rforest_rmse_scores,'RandomForestRegressor' )
#
# # KNN Regression
# n_neighbors=5
# knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
# knn.fit(housing_prepared,housing_labels)
# housing_predictions=knn.predict(housing_prepared)
# scoresKNeighborsRegressor=cross_val_score(knn, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# knn_rmse_scores = np.sqrt(-scoresKNeighborsRegressor)
# display_scores(knn_rmse_scores,'KNeighborsRegressor' )
#

# averaged_models = StackingAveragedModels(base_models = (forest_reg,knn,bayesian_ridge), meta_model= model_ridge)
# score = rmsle_cv_(averaged_models)
# print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


for i in range(len(models)) :
    averaged_models = StackingAveragedModels(base_models=(models[fabs(len(models)-i)],models[fabs(len(models)-1-i)],models[fabs(len(models)-2-i)],models[fabs(len(models)-3-i)],models[fabs(len(models)-4-i)],models[fabs(len(models)-5-i)],models[fabs(len(models)-6-i)],models[fabs(len(models)-7-i)],models[fabs(len(models)-8-i)],models[fabs(len(models)-9-i)],models[fabs(len(models)-10-i)] ), meta_model=models[fabs(len(models)-11-i)])
    score = rmsle_cv_(averaged_models)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
