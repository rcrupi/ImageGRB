# from gbm.finder import BurstCatalog
import pandas as pd
import numpy as np
from config.config import GBM_BURST_DB, GRB_REDSHIFT

# Load Burst GRB catalog of Fermi GBM
df_grb = pd.read_csv(GBM_BURST_DB)
print(df_grb.shape)
print(df_grb.head())

# Load data redshift GRB https://www.mpe.mpg.de/~jcg/grbgen.html
df_red = pd.read_csv(GRB_REDSHIFT, sep=';', encoding='latin')
# correct position for a bugged row
df_red.loc[df_red['GRBa'] == '190810AS', 'GRB X-ray position'] = "12h48m12sÿ+28ø 30'"
df_red['GRBa'] = df_red['GRBa'].apply(lambda x: x.replace('S', ''))
# 0 other S inside the name
if df_red.loc[df_red['GRBa'].str.contains('S'), 'GRBa'].str.slice(0, -1).str.contains('S').sum() > 0:
    print("warning: 'S' in name not at last.")


def redshift_clean(x):
    try:
        res = None
        if x == ' ' or x == '':
            res = None
        elif not pd.isna(x):
            if 'ph' in x:
                x = x.replace('ph', '')
            if '?' in x:
                x = x.replace('?', '')
            if '<' in x:
                x = x.replace('<', '')
            if '-' in x:
                x = x.split('-')
            if not isinstance(x, list):
                res = float(str(x).strip())
            else:
                res = (float(str(x[0]).strip()) + float(str(x[1]).strip()))/2
    except Exception as e:
        print(e)
        print('Failed to convert Redshift: ', x, "<---- what is this???")
        res = np.nan
    return res


# Clean the redshift values
df_red['z'] = df_red['zc'].apply(lambda x: redshift_clean(x))
df_red['GRB'] = df_red['GRBa']
# Get localization
df_red['ra'] = df_red['GRB X-ray position'].str.slice(0, 2).astype(int)/24*360
df_red['dec'] = df_red['GRB X-ray position'].apply(lambda x: x.split('ÿ')[1][0:3]).astype(int, errors='ignore')
# Selection only GRB name, redshift and localization
df_red = df_red[['GRB', 'z', 'ra', 'dec']]
# Define the "short" GRB name without the last letter or the last three number
df_grb['name_short'] = df_grb['name'].apply(lambda x: x[3:9])
df_red['name_short'] = df_red['GRB'].apply(lambda x: x[:-1] if x[-1].isalpha() else x)
# Merge the dataframes
df_tot_raw = df_grb.merge(df_red, on=['name_short'])
df_tot_raw_z = df_tot_raw[df_tot_raw['z'].notna()].copy()
# Define the difference in degree in localization
df_tot_raw_z.loc[:, 'diff_degree'] = abs(df_tot_raw_z['ra_x'] - df_tot_raw_z['ra_y']).values +\
                                     abs(df_tot_raw_z['dec_x'] - df_tot_raw_z['dec_y']).values
# Take the minimum degree difference
df_grb_diff_min = df_tot_raw_z.groupby('GRB', sort=True)['diff_degree'].min().reset_index()
# Choose the GRB with the miniumum degree difference
df_tot_z = df_grb_diff_min.merge(df_tot_raw_z, on=['GRB', 'diff_degree'], how='left')
# Selection only a maximum difference of 50°
df_tot_z = df_tot_z[df_tot_z['diff_degree'] < 50]
df_tot_z = df_tot_z.loc[df_tot_z.groupby('name').diff_degree.idxmin()].reset_index(drop=True)
# Duplicated events
print(df_tot_z.loc[df_tot_z.name.duplicated(keep=False),
                   ['name', 'trigger_time', 'diff_degree', 'ra_x', 'dec_x', 'ra_y', 'dec_y', 'z', 'GRB']])
print(df_tot_z.loc[df_tot_z.GRB.duplicated(keep=False),
                   ['name', 'trigger_time', 'diff_degree', 'ra_x', 'dec_x', 'ra_y', 'dec_y', 'z', 'GRB']])
print(df_tot_z.shape)

# # # Regression model
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import autosklearn.regression
import matplotlib.pyplot as plt
from tpot import TPOTRegressor

col_list = ['t90', 't90_error', 't50', 't50_error',
 'flux_64', 'flux_64_error', 'fluence', 'fluence_error',
 'pflx_band_phtfluxb', 'flnc_band_ampl', 'pflx_band_epeak', 'flnc_band_phtfluxb',
 'pflx_band_ergflux', 'flnc_band_phtflux', 'pflx_band_beta', 'flnc_band_ergflux',
 'pflx_band_phtflux', 'flnc_band_beta', 'flnc_band_epeak', 'pflx_band_ampl',
 'pflx_band_alpha', 'flnc_band_alpha']
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

combination_dataset = False
if combination_dataset:
    X_train['y'] = y_train
    X_train2 = pd.merge(X_train.assign(key=0), X_train.assign(key=0), how='left', on='key')
    del X_train['y']
    y_train2 = X_train2['y_x'] - X_train2['y_y']
    del X_train2['y_x'], X_train2['y_y']

    regr = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0) # max_features=10,
    regr.fit(X_train2, y_train2)
    y_pred_train2 = regr.predict(X_train2)


    def predict_row(regr, X_train, y_train, row):
        row = row.to_frame().T
        X_train2 = pd.merge(X_train.assign(key=0), row.assign(key=0), how='left', on='key')
        y_pred_row = -regr.predict(X_train2) + y_train
        return np.median(y_pred_row), np.std(y_pred_row)

    y_pred_test = []
    for index, row in X_test.iterrows():
        m, s = predict_row(regr, X_train, y_train, row)
        print(m, s)
        y_pred_test.append(m)

    y_pred_train = []
    for index, row in X_train.iterrows():
        m, s = predict_row(regr, X_train, y_train, row)
        print(m, s)
        y_pred_train.append(m)
else:
    # Model
    # regr = RandomForestRegressor(n_estimators=2000, max_depth=2, max_features=10, random_state=0)
    # regr = DecisionTreeRegressor(max_depth=5, random_state=0)
    # regr = autosklearn.regression.AutoSklearnRegressor(n_jobs=20)
    regr = TPOTRegressor(generations=100, population_size=100, offspring_size=None, mutation_rate=0.9,
                         crossover_rate=0.1, scoring=None, cv=5, subsample=1.0, n_jobs=-1, max_time_mins=None,
                         max_eval_time_mins=5, random_state=5, config_dict=None, template=None, warm_start=False,
                         memory=None, use_dask=False, periodic_checkpoint_folder=None, early_stop=None,
                         verbosity=2, disable_update_check=False, log_file=None)
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)

# print(pd.DataFrame({'col': X.columns, 'v': regr.feature_importances_}).sort_values(by='v', ascending=False))

fig = plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_test)
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_pred_train)
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.show()


# gredient boosting



all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=10,
    min_samples_leaf=9,
    min_samples_split=9,
)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

gbr_ls = GradientBoostingRegressor(loss="lad", **common_params)
all_models["mae"] = gbr_ls.fit(X_train, y_train)



y_pred = all_models["mae"].predict(X_train)
y_lower = all_models["q 0.05"].predict(X_train)
y_upper = all_models["q 0.95"].predict(X_train)
y_med = all_models["q 0.50"].predict(X_train)

y_pred_test = all_models["mae"].predict(X_test)
y_lower_test = all_models["q 0.05"].predict(X_test)
y_upper_test = all_models["q 0.95"].predict(X_test)
y_med_test = all_models["q 0.50"].predict(X_test)

fig = plt.figure(figsize=(10, 10))
plt.errorbar(y_test, y_med_test, yerr=[y_lower_test, y_upper_test], fmt="o")
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.errorbar(y_train, y_med, yerr=[y_lower, y_upper], fmt="o")
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.show()

pass