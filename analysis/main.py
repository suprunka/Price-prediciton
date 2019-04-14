import numpy as np
import pandas as pd
from matplotlib import *
import matplotlib.pyplot as plt
from pandas import DataFrame
import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
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
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression

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
virt['was_seen'] = virt['view'].apply(lambda x: 1 if x> 0 else 0)
t = virt.groupby('grade')[['price']]

#virt['yearsFromLastRenovation'] =  virt['yr_renovated'].apply(lambda x: int(now.year - x) if x > 0 else np.nan)
corr_matrix = virt.corr()


virt_num = virt


#Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = virt_num.quantile(0.25)
    Q3 = virt_num.quantile(0.75)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]




#prepare for machine learning
housing_labels = strat_train_set["price"].copy()

def add_additional_attributes(data):
    data['all_rooms'] = data['bathrooms'] + data['bedrooms']
    data.loc[data.all_rooms== 0, 'all_rooms'] = 1
    data['avg_room_size'] = data['sqm_living']/ data['all_rooms']
    data['avg_floor_sq'] = data['sqm_above'] / data['floors']
    data['was_seen'] = data['view'].apply(lambda x: 1 if x > 0 else 0)
    x = data[["zipcode", "price"]].groupby(['zipcode'], as_index=False).mean().sort_values(by='price',
                                                                                           ascending=False)
    x['zipcode_cat'] = 0
    x['zipcode_cat'] = np.where(x['price'] > 1000000, 3, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(750000, 1000000), 2, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(500000, 750000), 1, x['zipcode_cat'])
    # x=x.drop('price', axis=1)
    data = pd.merge(x, data, on=['zipcode'])
    return data
#NAJNOWSZE
def transform_data(data):
    data_filtered = get_rid_of_outliers(add_additional_attributes(data)[['sqm_basement', 'sqm_above', 'sqm_lot',
                                                                         'bedrooms', 'bathrooms', 'sqm_living', 'grade',
                                                                            ]])
    data_connected = pd.merge(data_filtered, data, how='inner', on=['sqm_basement', 'sqm_above', 'sqm_lot',
                                                                    'bedrooms', 'bathrooms', 'sqm_living',
                                                                    'grade'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_deleted_columns = data_no_duplicates.drop(['bathrooms', 'bedrooms', 'floors'], axis=1)
    cols = [col for col in data_deleted_columns.columns if col not in ['price', 'id']]
    data_scaled = StandardScaler().fit_transform(data_no_duplicates[cols])
    return data_scaled


def get_labels(data):
    data_filtered = get_rid_of_outliers(add_additional_attributes(data)[['sqm_basement', 'sqm_above', 'sqm_lot',
                                                                         'bedrooms', 'bathrooms','sqm_living', 'grade']])
    data_connected = pd.merge(data_filtered, data, how='inner',on=['sqm_basement', 'sqm_above', 'sqm_lot',
                                                                   'bedrooms','bathrooms', 'sqm_living','grade'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_prepared = data_no_duplicates['price']
    to_return = data_prepared.values
    return to_return


housing_prepared = transform_data(housing)
housing_labels = get_labels((housing))



#Select and train a model

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




# forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# scoresRandomForestRegression =cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rforest_rmse_scores = np.sqrt(-scoresRandomForestRegression)
# display_scores(rforest_rmse_scores)




# svm_reg = SVR(kernel="rbf")
# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# scoresSVR =cross_val_score(svm_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# svr_scores = np.sqrt(-scoresSVR)
# display_scores(svr_scores)




def randmizedSearchCV():
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    return rnd_search.cv_results_

#TODO pozbyć się problemu z get_dummies
#TODO obecniy wynik to 88842 z randmizedSearchCV
#TODO spawdzić inne modele
#TODO naprawić problem z lasso
#TODO manualne usuwanie outlierów
#TODO stworzenie nowych zmiennych które mogą mieć wpływ


def gridSearchCV():
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    return grid_search.cv_results_

def lassoSearch():
    lasso = Lasso(max_iter=10000, normalize=True)
    lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
    lassocv.fit(housing_prepared, housing_labels)
    lasso.set_params(alpha=lassocv.alpha_)
    print("Alpha=", lassocv.alpha_)
    lasso.fit(housing_prepared, housing_labels)
    print("mse = ", mean_squared_error(transform_data(strat_test_set), get_labels(strat_test_set)))
    print("best model coefficients:")
    pd.Series(lasso.coef_, index=housing.columns)



cvres = randmizedSearchCV()
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


