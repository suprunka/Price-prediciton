import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn import linear_model
from sklearn import neighbors
from scipy import stats
from scipy.special import boxcox1p
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
#AKTUALIZACJA LOL


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

#Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.25)
    Q3 = num_data.quantile(0.75)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]




def add_additional_attributes(data):
    data['all_rooms'] = data['bathrooms'] + data['bedrooms']
    data.loc[data.all_rooms== 0, 'all_rooms'] = 1
    data['avg_room_size'] = data['sqm_living']/ data['all_rooms']
    data['avg_floor_sq'] = data['sqm_above'] / data['floors']
    data['was_seen'] = data['view'].apply(lambda x: 1 if x > 0 else 0)
    x = data[["zipcode", "price"]].groupby(['zipcode'], as_index=False).mean().sort_values(by='price', ascending=False)
    x['zipcode_cat'] = 0
    x['zipcode_cat'] = np.where(x['price'] > 1000000, 3, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(750000, 1000000), 2, x['zipcode_cat'])
    x['zipcode_cat'] = np.where(x['price'].between(500000, 750000), 1, x['zipcode_cat'])
    x= x.drop('price', axis=1)
    data = pd.merge(x, data, on=['zipcode'])
    data['avg_room_size'] = stats.boxcox(np.asanyarray(data[['avg_room_size']].values))[0]
    data['sqm_above'] = stats.boxcox(np.asanyarray(data[['sqm_above']].values))[0]
    data['sqm_lot'] = stats.boxcox(np.asanyarray(data[['sqm_lot']].values))[0]
    data['sqm_living'] = stats.boxcox(np.asanyarray(data[['sqm_living']].values))[0]
    return data
    #73529.99236662195

def transform_data(data):
    data_additional_attributes = add_additional_attributes(data)
    data_filtered = get_rid_of_outliers(data_additional_attributes)[['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner', on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_deleted_columns = data_no_duplicates.drop(['bathrooms', 'zipcode', 'view', 'bedrooms', 'floors'], axis=1)
    cols = [col for col in data_deleted_columns.columns if col not in ['price', 'id']]
    data_scaled = MinMaxScaler().fit_transform(data_no_duplicates[cols])
    return data_scaled


def get_labels(data):
    data_additional_attributes = add_additional_attributes(data)
    data_filtered = get_rid_of_outliers(data_additional_attributes)[['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms', 'bathrooms','sqm_living']]
    data_connected = pd.merge(data_filtered, data_additional_attributes, how='inner',on=['sqm_basement', 'sqm_above', 'sqm_lot', 'bedrooms','bathrooms', 'sqm_living'])
    data_no_duplicates = data_connected.drop_duplicates(['id'])
    data_prepared = data_no_duplicates['price']
    to_return = data_prepared.values
    return to_return


housing_prepared = transform_data(housing)
housing_labels = get_labels((housing))



# #Select and train a model
#
def display_scores(scores, model):
    print("#####", model, "#####")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


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

    parameters = {'n_neighbors':[4,5,6,7], 'leaf_size':[1,3,5],
                  'algorithm':['auto', 'kd_tree'], 'n_jobs':[-1]}

    model = GridSearchCV(knn, param_grid=parameters)
    grid_result = model.fit(housing_prepared, housing_labels)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)


# LinearRegression
lin_reg = linear_model.LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                        scoring="neg_mean_squared_error", cv=10)
linear_scores = np.sqrt(-scores)
display_scores(linear_scores, 'Linear Regression')

# BayesianRidge
bayesian_ridge = linear_model.BayesianRidge()
bayesian_ridge.fit(housing_prepared, housing_labels)
housing_predictions = bayesian_ridge.predict(housing_prepared)
bayesian_ridge_scores = cross_val_score(bayesian_ridge, housing_prepared, housing_labels,
                                        scoring="neg_mean_squared_error", cv=10)
br_scores = np.sqrt(-bayesian_ridge_scores)
display_scores(br_scores, 'Bayesian Ridge')

# Ridge
model_ridge = linear_model.Ridge()
model_ridge.fit(housing_prepared, housing_labels)
housing_predictions = model_ridge.predict(housing_prepared)
scores = cross_val_score(model_ridge, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
model_ridge_rmse = np.sqrt(-scores)
display_scores(model_ridge_rmse, 'Ridge Model')



# DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores, 'Decision Tree Regressorr')



# RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=50, random_state=42, max_features="auto", max_depth=100,
                                   min_samples_leaf=4, bootstrap=True, min_impurity_decrease=100)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
scoresRandomForestRegression = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                               scoring="neg_mean_squared_error", cv=10)
rforest_rmse_scores = np.sqrt(-scoresRandomForestRegression)
display_scores(rforest_rmse_scores, 'Random Forest Regressor')



# KNN Regression

knn = neighbors.KNeighborsRegressor(n_neighbors= 7, weights='uniform', algorithm="auto", leaf_size=1, n_jobs=-1)
knn.fit(housing_prepared, housing_labels)
housing_predictions = knn.predict(housing_prepared)
scoresKNeighborsRegressor = cross_val_score(knn, housing_prepared, housing_labels, scoring="neg_mean_squared_error",
                                            cv=10)
knn_rmse_scores = np.sqrt(-scoresKNeighborsRegressor)
display_scores(knn_rmse_scores, 'KNeighborsRegressor')


