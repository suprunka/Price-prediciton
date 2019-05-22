from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def gridSearchCV(X, Y):
    pip_lin_reg = Pipeline((('clf', LinearRegression()),))
    pip_bayesian_ridge = Pipeline((('clf', linear_model.BayesianRidge()),))
    pip_ridge = Pipeline((('clf', linear_model.Ridge()),))
    pip_tree_reg = Pipeline((('clf', DecisionTreeRegressor()),))
    pip_random_forest_reg = Pipeline((('clf', RandomForestRegressor()),))
    pip_knn = Pipeline((('clf', KNeighborsRegressor()),))
    pip_lasso = Pipeline((('clf', Lasso()),))
    pip_elastic = Pipeline((('clf', ElasticNet()),))
    pip_gradient_boost = Pipeline((('clf',GradientBoostingRegressor()),))
    pip_xgbr = Pipeline((('clf', xgb.XGBRegressor()),))
    pip_bagreg = Pipeline((('clf', BaggingRegressor()),))

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

    parameters9 = {'clf__loss': ['ls', 'lad', 'huber', 'quantile'],
                   'clf__learning_rate': [0.01, 0.1, 1],
                   'clf__n_estimators': [50, 100, 150],
                   'clf__subsample': [0.1, 0.5, 1],
                   'clf__min_samples_split': [2, 4,8],
                   'clf__min_samples_leaf': [1, 2, 4],
                   'clf__max_depth':[3, 6, 9, 15],
                   'clf__alpha': [0.3, 0.5, 0.9]}

    parameters10 = {'clf__booster': ['gbtree', 'gblinear', 'dart'],
                    'clf__verbosity': [0,1,2,3],
                    'clf__nthreads': [10, 50, 100]}

    parameters11 = {'clf__n_estimators': [5, 10 ,20, 50],
                    'clf__bootstrap': [True, False],
                    'clf__bootstrap_features': [True, False],
                    'clf__warm_start': [True,False],
                    'clf__n_jobs': [None, 2, 5]}

    pips = [pip_xgbr]
    pars = [parameters10]

    print("Starting Gridsearch")
    for i in range(len(pars)):
        gs = RandomizedSearchCV(pips[i], pars[i], verbose=2, refit=False, n_jobs=-1)
        gs = gs.fit(X, Y)
        print("Best parameters of: ", i.__class__ , " ", gs.best_params_)

