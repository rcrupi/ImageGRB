import numpy as np
import pandas as pd
from config.config import GBM_BURST_DB, GRB_REDSHIFT


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


def build_dataset_gbm_z():

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
    return df_tot_z
