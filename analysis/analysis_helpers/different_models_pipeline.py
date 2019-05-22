#
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge, LinearRegression # Linear models
# from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
# from xgboost import XGBRegressor, plot_importance # XGBoost
# from sklearn.svm import SVR, SVC, LinearSVC  # Support Vector Regression
# from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
# from sklearn.neighbors import KNeighborsRegressor
# import lightgbm as lgb
# from sklearn.pipeline import Pipeline # Streaming pipelines
# from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction
# from sklearn.feature_selection import SelectFromModel # Dimensionality reduction
# from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV # Model evaluation
# from sklearn.base import clone # Clone estimator
# from sklearn.metrics import mean_squared_error as MSE
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# def create_pipeline(train_x, train_y, seed=2):
#     pipelines = []
#
#
#     pipelines.append(
#         ("Scaled_Ridge",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("Ridge", Ridge(random_state=seed, tol=10))
#          ]))
#     )
#
#     pipelines.append(
#         ("LGBNRegressor",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("LGBNRegressor", lgb.LGBMRegressor(random_state=seed))
#          ]))
#
#     )
#
#     pipelines.append(
#         ("Linear Regression",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("Linear Regression", LinearRegression())
#          ]))
#     )
#
#     pipelines.append(
#         ("Scaled_Lasso",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("Lasso", Lasso(random_state=seed, tol=1))
#          ]))
#     )
#     pipelines.append(
#         ("Scaled_Elastic",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("Lasso", ElasticNet(random_state=seed))
#          ]))
#     )
#
#     pipelines.append(
#         ("Scaled_SVR",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("SVR", SVR(kernel='linear', C=1e2, degree=5))
#          ])
#          )
#     )
#
#     pipelines.append(
#         ("Scaled_RF_reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("RF", RandomForestRegressor(random_state=seed))
#          ])
#          )
#     )
#
#     pipelines.append(
#         ("Scaled_ET_reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("ET", ExtraTreesRegressor(random_state=seed))
#          ])
#          )
#     )
#     pipelines.append(
#         ("Scaled_BR_reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("BR", BaggingRegressor(random_state=seed))
#          ])))
#
#     pipelines.append(
#         ("Scaled_Hub-Reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("Hub-Reg", HuberRegressor())
#          ])))
#     pipelines.append(
#         ("Scaled_BayRidge",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("BR", BayesianRidge())
#          ])))
#
#     pipelines.append(
#         ("Scaled_XGB_reg",
#          Pipeline([
#              ("XGBR", XGBRegressor(seed=seed, n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7, nthread=100))
#          ])))
#
#     pipelines.append(
#         ("Scaled_DT_reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("DT_reg", DecisionTreeRegressor())
#          ])))
#
#     pipelines.append(
#         ("Scaled_KNN_reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("KNN_reg", KNeighborsRegressor())
#          ])))
#
#     pipelines.append(
#         ("Scaled_ADA-Reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("ADA-reg", AdaBoostRegressor())
#          ])))
#
#     pipelines.append(
#         ("Scaled_Gboost-Reg",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("GBoost-Reg", GradientBoostingRegressor())
#          ])))
#
#     pipelines.append(
#         ("Scaled_RFR_PCA",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("PCA", PCA(n_components=3)),
#              ("XGB", RandomForestRegressor())
#          ])))
#
#     pipelines.append(
#         ("Scaled_XGBR_PCA",
#          Pipeline([
#              ("Scaler", StandardScaler()),
#              ("PCA", PCA(n_components=3)),
#              ("XGB", XGBRegressor())
#          ])))
#
#     results, names = [], []
#
#     for name, model in pipelines:
#         kfold = KFold(n_splits=7, random_state=2)
#         cv_results = cross_val_score(model, train_x, train_y, cv=kfold,
#                                      scoring='r2', n_jobs=-1)
#         names.append(name)
#         results.append(cv_results)
#         msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
#         print(msg)
#
#
#     # boxplot algorithm comparison
#     fig = plt.figure(figsize=(15, 6))
#     fig.suptitle('Algorithm Comparison', fontsize=22)
#     ax = fig.add_subplot(111)
#     sns.boxplot(x=names, y=results)
#     ax.set_xticklabels(names)
#     ax.set_xlabel("Algorithmn Name", fontsize=20)
#     ax.set_ylabel("R Squared Score of Models", fontsize=18)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     plt.show()