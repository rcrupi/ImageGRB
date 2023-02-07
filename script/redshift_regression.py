# from gbm.finder import BurstCatalog
import pandas as pd
import numpy as np
from utils.read_data import build_dataset_gbm_z

# Build the merged dataset
df_tot_z = build_dataset_gbm_z()

# # # Regression model
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import autosklearn.regression
import matplotlib.pyplot as plt
from tpot import TPOTRegressor

# col_list = ['t90', 't90_error', 't50', 't50_error',
#  'flux_64', 'flux_64_error', 'fluence', 'fluence_error',
#  'pflx_band_phtfluxb', 'flnc_band_ampl', 'pflx_band_epeak', 'flnc_band_phtfluxb',
#  'pflx_band_ergflux', 'flnc_band_phtflux', 'pflx_band_beta', 'flnc_band_ergflux',
#  'pflx_band_phtflux', 'flnc_band_beta', 'flnc_band_epeak', 'pflx_band_ampl',
#  'pflx_band_alpha', 'flnc_band_alpha']
col_list = list(df_tot_z.columns[df_tot_z.dtypes == 'float64']) + ['dec_y']
col_list = [i for i in col_list if i != 'z']
X = df_tot_z[col_list].copy()
y = df_tot_z['z']
# imputer = KNNImputer(n_neighbors=2, weights="uniform")
# X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X = X.fillna(-1)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.1, random_state=0, shuffle=True # False
   )

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.1, random_state=0, shuffle=True # False
   )

combination_dataset = True
if combination_dataset:
    # X train
    X_train['y'] = y_train
    X_train2 = pd.merge(X_train.assign(key=0), X_train.assign(key=0), how='left', on='key')
    del X_train['y']
    y_train2 = X_train2['y_x'] - X_train2['y_y']
    del X_train2['y_x'], X_train2['y_y']
    # X val
    X_train['y'] = y_train
    X_val['y'] = y_val
    X_val2 = pd.merge(X_train.assign(key=0), X_val.assign(key=0), how='left', on='key')
    del X_val['y'], X_train['y']
    y_val2 = X_val2['y_x'] - X_val2['y_y']
    del X_val2['y_x'], X_val2['y_y']

    # regr = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1) # max_features=10,
    regr = ExtraTreesRegressor(n_estimators=500, max_depth=8, random_state=0, n_jobs=-1)
    #regr = TPOTRegressor(generations=5, population_size=20, n_jobs=-1, random_state=5)

    # lst_feat_added = []
    # lst_max_score = []
    # for i in range(0, 5):
    #     print('iteration', i)
    #     max_score = 0
    #     feat_to_add = None
    #     for j in X_train2.columns:
    #         if j in lst_feat_added:
    #             continue
    #         regr = ExtraTreesRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)
    #         lst_feat_added_tmp = lst_feat_added + [j]
    #         regr.fit(X_train2.loc[:, lst_feat_added_tmp], y_train2)
    #         max_tmp = regr.score(X_val2.loc[:, lst_feat_added_tmp], y_val2)
    #         if max_score <= max_tmp:
    #             max_score = max_tmp
    #             feat_to_add = j
    #     if feat_to_add is None:
    #         feat_to_add = [i for i in X_train2.columns if i not in lst_feat_added][0]
    #     print("feature added", feat_to_add)
    #     lst_max_score = lst_max_score + [max_score]
    #     lst_feat_added = lst_feat_added + [feat_to_add[:-1]+'x', feat_to_add[:-1]+'y']

    regr.fit(X_train2, y_train2)
    y_pred_train2 = regr.predict(X_train2)


    def predict_row(regr, X_train, y_train, row, k=1):
        row = row.to_frame().T
        X_train2 = pd.merge(X_train.assign(key=0), row.assign(key=0), how='left', on='key')
        y_pred2 = regr.predict(X_train2)
        argmin_pred = np.argsort(abs(y_pred2))[0:k]
        y_pred_row = -y_pred2[argmin_pred] + y_train.iloc[argmin_pred]
        return np.mean(y_pred_row), np.std(y_pred_row)

    y_pred_test = []
    for index, row in X_test.iterrows():
        m, s = predict_row(regr, X_train, y_train, row)
        # print(m, s)
        y_pred_test.append(m)

    y_pred_train = []
    for index, row in X_train.iterrows():
        m, s = predict_row(regr, X_train, y_train, row)
        # print(m, s)
        y_pred_train.append(m)
else:
    # Model
    # regr = RandomForestRegressor(n_estimators=2000, max_depth=2, max_features=10, random_state=0)
    # regr = DecisionTreeRegressor(max_depth=5, random_state=0)
    # regr = autosklearn.regression.AutoSklearnRegressor(n_jobs=20)
    # regr = ExtraTreesRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)
    regr = TPOTRegressor(generations=100, population_size=100, offspring_size=None, mutation_rate=0.9,
                         crossover_rate=0.1, scoring=None, cv=5, subsample=1.0, n_jobs=-1, max_time_mins=None,
                         max_eval_time_mins=5, random_state=5, config_dict=None, template=None, warm_start=False,
                         memory=None, use_dask=False, periodic_checkpoint_folder=None, early_stop=None,
                         verbosity=2, disable_update_check=False, log_file=None)
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)

if "feature_importances_" in dir(regr):
    if combination_dataset:
        print(pd.DataFrame({'col': X_train2.columns, 'v': regr.feature_importances_}).sort_values(by='v', ascending=False))
    else:
        print(pd.DataFrame({'col': X.columns, 'v': regr.feature_importances_}).sort_values(by='v', ascending=False))

fig = plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_pred_train, label='train')
plt.scatter(y_test, y_pred_test, label='test')
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.legend()
plt.show()


# # gredient boosting
# all_models = {}
# common_params = dict(
#     learning_rate=0.05,
#     n_estimators=2000,
#     max_depth=10,
#     min_samples_leaf=9,
#     min_samples_split=9,
# )
# for alpha in [0.05, 0.5, 0.95]:
#     gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
#     all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
#
# gbr_ls = GradientBoostingRegressor(loss="lad", **common_params)
# all_models["mae"] = gbr_ls.fit(X_train, y_train)
#
#
#
# y_pred = all_models["mae"].predict(X_train)
# y_lower = all_models["q 0.05"].predict(X_train)
# y_upper = all_models["q 0.95"].predict(X_train)
# y_med = all_models["q 0.50"].predict(X_train)
#
# y_pred_test = all_models["mae"].predict(X_test)
# y_lower_test = all_models["q 0.05"].predict(X_test)
# y_upper_test = all_models["q 0.95"].predict(X_test)
# y_med_test = all_models["q 0.50"].predict(X_test)
#
# fig = plt.figure(figsize=(10, 10))
# plt.errorbar(y_test, y_med_test, yerr=[y_lower_test, y_upper_test], fmt="o")
# plt.plot([0,8],[0,8])
# plt.xlabel("true")
# plt.ylabel("predicted")
# plt.show()
#
# fig = plt.figure(figsize=(10, 10))
# plt.errorbar(y_train, y_med, yerr=[y_lower, y_upper], fmt="o")
# plt.plot([0,8],[0,8])
# plt.xlabel("true")
# plt.ylabel("predicted")
# plt.show()

pass