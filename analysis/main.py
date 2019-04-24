import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from dbConnection import *
import math


# Where to save the figures
PROJECT_ROOT_DIR = r"D:\\"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


now = datetime.datetime.now()


housing_n = get_data()
# Constant values
valueOfSqM = 10.76
numberOfBins = 4
divider = 1000
threshold = 3
low = .05
high = .95


# <editor-fold desc='Dropping unnecessary columns'>
housing_n = housing_n.drop(["sqft_living15", "sqft_lot15", 'date', 'waterfront'], axis=1)
housing_n['floors'] = housing_n['floors'].str[1:-1]
housing_n['zipcode'] = housing_n['zipcode'].str[1:-1]
housing = housing_n.convert_objects(convert_numeric=True)
# housing = housing[:-1]
# </editor-fold>
# <editor-fold desc='Creating columns which represent data in square meters'>
housing['sqm_living'] = round(housing['sqft_living']/valueOfSqM)
housing['sqm_lot'] = round(housing['sqft_lot']/valueOfSqM)
housing['sqm_above'] = round(housing['sqft_above']/valueOfSqM)
housing['sqm_basement'] = round(housing['sqft_basement']/valueOfSqM)
# </editor-fold>
# <editor-fold desc='Dropping columns which represent data in square foots'>
housing = housing.drop(["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], axis=1)
# </editor-fold>

# Creating a price category
maximum = housing['price'].max()/divider
minimum = housing['price'].min()/divider

# Creating bins using the price category


def create_bins(maximum, minimum):
    x = [maximum+1]
    difference = (maximum-minimum)/numberOfBins
    for i in range(numberOfBins):
        if i == 0:
            x.append(int((minimum-1 +(difference*i))))
        else:
            x.append(int((minimum + (difference * i))))
    x.sort()
    return x


housing['price_cat'] = pd.cut(housing['price']/divider, bins= create_bins(maximum, minimum), labels = [1,2,3,4])

# Splitting data in train and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["price_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Deleting price_category from train and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)


housing = strat_train_set.copy()
# Creating copy of dataset for virtualization
virt = housing.copy()
virt['all_rooms'] = virt['bathrooms'] + virt['bedrooms']
virt.loc[virt.all_rooms== 0, 'all_rooms'] = 1
virt['avg_room_size'] = virt['sqm_living']/ virt['all_rooms']
virt['avg_floor_sq'] = virt['sqm_above'] / virt['floors']
virt['was_seen'] = virt.loc[virt.view > 0, 'was_seen'] = 1

corr_matrix = virt.corr()

virt_num = virt


# Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.3)
    Q3 = num_data.quantile(0.7)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]


# prepare for machine learning
# housing = strat_train_set.drop(['id'], axis=1)
housing_labels = strat_train_set["price"].copy()

# Get only numeric values
# housing_num = strat_train_set.drop("date", axis=1)
# Distinguish housing categories for processing : get year and month


# def transform_categorital(data):
#     data_cat = pd.get_dummies(data['date'].apply(lambda x: x.replace(x, x[0:6])), prefix = 'date')
#     return_data = pd.concat([data, data_cat], axis=1)
#     return return_data.drop('date',axis=1)

def add_additional_attributes(data):
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
    x=x.drop('price', axis=1)
    data = pd.merge(x, data, how='right', on=['zipcode'])
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
    data_filtered = get_rid_of_outliers(data_additional_attributes)[
        ['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner',
                              on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_deleted_columns = data_no_duplicates.drop(['bathrooms', 'yr_renovated','view', 'bedrooms', 'floors',
                                                    'condition'], axis=1)
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
housing_labels = get_labels(housing)
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




def gridSearchCV(X, Y):
    pip_lin_reg = Pipeline((('clf', LinearRegression()),))
    pip_bayesian_ridge = Pipeline((('clf', linear_model.BayesianRidge()),))
    pip_ridge = Pipeline((('clf', linear_model.Ridge()),))
    pip_tree_reg = Pipeline((('clf', DecisionTreeRegressor()),))
    pip_random_forest_reg = Pipeline((('clf', RandomForestRegressor()),))
    pip_knn = Pipeline((('clf', KNeighborsRegressor()),))
    pip_lasso = Pipeline((('clf', Lasso()),))
    pip_elastic = Pipeline((('clf', ElasticNet()),))

    parameters1 = {'clf__fit_intercept': [True, False],
                   'clf__normalize': [True, False],
                   'clf__copy_X': [True,False],
                   'clf__n_jobs': [-1, None, 1, 2, 3]}

    parameters2 = {'clf__n_iter':[100, 200, 300, 400, 500],
                   'clf__tol': [0.001, 0.01, 0.1, 1],
                   'clf__alpha_1': [0.000001, 0.00001, 0.0001],
                   'clf__alpha_2': [0.000001, 0.00001, 0.0001]}

    parameters3 = {'clf__alpha':[1.0, 10.0, 50.0, 100.0],
                   'clf__fit_intercept': [True, False],
                   'clf__normalize': [True, False],
                   'clf__copy_X': [True, False],
                   'clf__max_iter': [None, 10, 100, 1000]}

    parameters4 = {'clf__splitter': ['best', 'random'],
                   'clf__max_depth': [None, 1, 5, 10, 50, 100, 500, 1000],
                   'clf__min_samples_split': [2, 4, 8, 16, 32, 64],
                   'clf__min_samples_leaf': [1,2,4,8,10,50,100],
                   'clf__min_weight_fraction_leaf': [0., .2, .5],
                   'clf__presort': [True, False],
                   'clf__min_impurity_decrease':[0.,0.1,.5,.9]}

    parameters5 = {'clf__n_estimators': [10, 100],
                   'clf__max_depth': [50,100,150,200],
                   'clf__min_samples_split': [16,32,64,],
                   'clf__min_samples_leaf':[8,10,50],
                   'clf__min_weight_fraction_leaf': [0.,.5],
                   'clf__min_impurity_decrease': [0.,.1,.5,.9],
                   'clf__n_jobs': [None,-1, 1]}

    parameters6 = {'clf__n_neighbors': [5,10,20],
                   'clf__weights': ['uniform', 'distance'],
                   'clf__algorithm':['auto','ball_tree','kd_tree','brute'],
                   'clf__leaf_size':[30,90,180,360],
                   'clf__n_jobs': [None, -1, 1]}

    parameters7 = {'clf__alpha': [0., 1., 5.,10.],
                   'clf__fit_intercept': [True, False],
                   'clf__normalize': [True, False],
                   'clf__precompute': [True, False],
                   'clf__copy_X': [True,False],
                   'clf__max_iter': [10, 100, 1000, 10000],
                   'clf__tol': [0.0001, 0.001, 0.01, 0.1],
                   'clf__warm_start': [True, False],
                   'clf__positive':[True, False],
                   'clf__selection': ['random', 'cyclic']}

    parameters8 = {'clf__alpha': [0., 1., 5., 0.],
                    'clf__l1_ratio': [0.1,0.5,0.9],
                    'clf__fit_intercept': [False, True],
                    'clf__normalize': [True, False],
                    'clf__precompute':[True, False],
                    'clf__max_iter': [10, 100, 1000],
                    'clf__copy_X': [True, False],
                    'clf__tol': [0.0001, 0.001, 0.01],
                    'clf__warm_start':[True, False],
                    'clf__positive': [True, False],
                    'clf__selection': ['cyclic', 'random']}
    pips = [pip_elastic]
    pars = [parameters8]

    print("Starting Gridsearch")
    for i in range(len(pars)):
        gs = RandomizedSearchCV(pips[i], pars[i], verbose=2, refit=False, n_jobs=-1)
        gs = gs.fit(X, Y)
        print("Best parameters of: ", i.__class__ ," ",gs.best_params_)




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

#
# n_estimators=50, random_state=42, max_features="auto", max_depth=100,
#                                    min_samples_leaf=4, bootstrap=True, min_impurity_decrease=100
lin_reg = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1,normalize=True)



bayesian_ridge = linear_model.BayesianRidge(n_iter=100, tol=0.01, alpha_1=0.0001, alpha_2=0.000001)

model_ridge=linear_model.Ridge(alpha=10, copy_X=True, fit_intercept=False, max_iter=None, normalize=True)

tree_reg = DecisionTreeRegressor(max_depth=500, min_impurity_decrease=0.5, min_samples_leaf=2, min_samples_split=64,
                                 min_weight_fraction_leaf=0.0, presort=False, splitter='random')

forest_reg =  RandomForestRegressor(n_jobs= -1, min_weight_fraction_leaf=0., n_estimators=100, min_samples_split=16,
                                    min_samples_leaf=8, min_impurity_decrease=.5, max_depth=50)

knn=neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, n_jobs=None, leaf_size=360,
                                   algorithm='ball_tree')

lasso =  Lasso(warm_start=False, tol=0.01, selection='random', precompute=True,
                                            positive=False,normalize=False, max_iter=10, fit_intercept=False,
                                            copy_X=True, alpha=5.0)
#lepszy wynik bez make_pipeline dla lasso i eNet
ENet = ElasticNet(warm_start=False, tol=0.001, selection='random',
                                                precompute=True, positive=False, normalize=False,
                                                max_iter=1000, l1_ratio=0.1, fit_intercept=False,
                                                copy_X=True, alpha=0.)

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


def stacking_avg_for_all_combinations_of_models(models):
    for i in range(len(models)) :
        averaged_models = StackingAveragedModels(base_models=(models[math.fabs(len(models)-i)],
        models[math.fabs(len(models)-1-i)],models[math.fabs(len(models)-2-i)],models[math.fabs(len(models)-3-i)],
        models[math.fabs(len(models)-4-i)],models[math.fabs(len(models)-5-i)],models[math.fabs(len(models)-6-i)],
        models[math.fabs(len(models)-7-i)],models[math.fabs(len(models)-8-i)],models[math.fabs(len(models)-9-i)],
        models[math.fabs(len(models)-10-i)] ), meta_model=models[math.fabs(len(models)-11-i)])
        score = rmsle_cv_(averaged_models)
        print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

models = [forest_reg ,GBoost, model_xgb,  model_lgb]
checkAllModels(models)
#stacking_avg_for_all_combinations_of_models(models)
#averaged_models = StackingAveragedModels(base_models=(GBoost, model_xgb,  model_lgb), meta_model =forest_reg)
#score = rmsle_cv_(averaged_models)
#print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# gridSearchCV(housing_prepared, housing_labels)
# pickle.dump(forest_reg, open('model.pkl', 'wb'))
