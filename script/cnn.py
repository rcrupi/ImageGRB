# Standard package
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
import matplotlib.cm
# from PIL import Image
import pandas as pd
import pickle
from tqdm import tqdm
# local import
from config.config import PATH_GRB_PKL, PATH_GRB_IMG
from utils.read_data import build_dataset_gbm_z
# nn
import tensorflow as tf

np.random.seed(42)

df_tot_z = build_dataset_gbm_z()

LEN_LC=512
n_load=None
bln_save_img_scale=True
bln_augment=False
scaler='Standard'
cmap='Greys_r'
min_amp=5.0
norm=False

lst_tte_pkl_aug = []
np.random.seed(10)
counter_set = 0

counter_constant = 0
# List of all GRB background subtracted and saved as a 2D array
list_tte_pkl = [i for i in os.listdir(PATH_GRB_PKL) if 'pickle' in i if i[2:11] in df_tot_z['name'].str.slice(3).to_list()]
n_ev = len(list_tte_pkl)
list_tte_pkl_train = np.random.choice(list_tte_pkl, int(n_ev*0.8), replace=False)
list_tte_pkl_residual = [i for i in list_tte_pkl if i not in list_tte_pkl_train]
n_ev_res = len(list_tte_pkl_residual)
list_tte_pkl_val = np.random.choice(list_tte_pkl_residual, int(n_ev_res*0.5), replace=False)
list_tte_pkl_test = [i for i in list_tte_pkl_residual if i not in list_tte_pkl_val]


def load_set(list_tte_pkl):
    lst_tte_pkl_aug = []
    ds_train = []
    for i in tqdm(list_tte_pkl):
        with open(PATH_GRB_PKL + i, 'rb') as f:
            event_tmp = pickle.load(f)
            if abs(event_tmp.max() - event_tmp.min()) < 10 ** -6:
                print("event constant.")
                continue
            ds_train.append(event_tmp)
            lst_tte_pkl_aug.append(' '.join(i.split('_')) + " " + "no shift")
            if bln_augment:
                event_tmp_r_shift = np.roll(event_tmp, np.random.randint(0, int(LEN_LC / 4), 1)[0], axis=0)
                ds_train.append(event_tmp_r_shift)
                lst_tte_pkl_aug.append(' '.join(i.split('_')) + " " + "right shift")
                event_tmp_l_shift = np.roll(event_tmp, -np.random.randint(0, int(LEN_LC / 4), 1)[0], axis=0)
                ds_train.append(event_tmp_l_shift)
                lst_tte_pkl_aug.append(' '.join(i.split('_')) + " " + "left shift")
    ds_train = np.array(ds_train)
    return ds_train, lst_tte_pkl_aug

ds_train, lst_train = load_set(list_tte_pkl_train)
ds_test, lst_test = load_set(list_tte_pkl_test)
ds_val, lst_val = load_set(list_tte_pkl_val)
y_train = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], 'z'].values[0] for i in lst_train])
y_test = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], 'z'].values[0] for i in lst_test])
y_val = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], 'z'].values[0] for i in lst_val])

col_list = ['t90', 't90_error', 't50', 't50_error',
 'flux_64', 'flux_64_error', 'fluence', 'fluence_error',
 'pflx_band_phtfluxb', 'flnc_band_ampl', 'pflx_band_epeak', 'flnc_band_phtfluxb',
 'pflx_band_ergflux', 'flnc_band_phtflux', 'pflx_band_beta', 'flnc_band_ergflux',
 'pflx_band_phtflux', 'flnc_band_beta', 'flnc_band_epeak', 'pflx_band_ampl',
 'pflx_band_alpha', 'flnc_band_alpha']

df_tot_z.loc[:, col_list] = df_tot_z.loc[:, col_list].fillna(-1)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
scaler = QuantileTransformer()
add_train = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], col_list].values[0] for i in lst_train])
add_train = scaler.fit_transform(add_train)
add_test = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], col_list].values[0] for i in lst_test])
add_test = scaler.transform(add_test)
add_val = np.array([df_tot_z.loc[df_tot_z['name'].str.slice(3) == i[2:11], col_list].values[0] for i in lst_val])
add_val = scaler.transform(add_val)

# mapper from 2d array to rgba image
sm = matplotlib.cm.ScalarMappable(cmap)

# Scale dataset

ds_train_scale = ds_train.copy()
for i in tqdm(range(0, ds_train.shape[0])):
    # Standard scaler
    scaler = StandardScaler()
    ds_train_scale[i, :, :] = scaler.fit_transform(ds_train[i, :, :])
    ds_train_max = ds_train_scale[i, :, :].max()
    ds_train_min = ds_train_scale[i, :, :].min()
    if abs(ds_train_max - ds_train_min) > 10 ** -6:
        ds_train_scale[i, :, :] = (ds_train_scale[i, :, :] - ds_train_min) / \
                                  (max(min_amp, ds_train_max) - ds_train_min)

ds_test_scale = ds_test.copy()
for i in tqdm(range(0, ds_test.shape[0])):
    # Standard scaler
    ds_test_scale[i, :, :] = scaler.transform(ds_test[i, :, :])
    ds_test_max = ds_train_scale[i, :, :].max()
    ds_test_min = ds_train_scale[i, :, :].min()
    if abs(ds_test_max - ds_test_min) > 10 ** -6:
        ds_test_scale[i, :, :] = (ds_test_scale[i, :, :] - ds_test_min) / \
                                  (max(min_amp, ds_test_max) - ds_test_min)

ds_val_scale = ds_val.copy()
for i in tqdm(range(0, ds_val.shape[0])):
    # Standard scaler
    ds_val_scale[i, :, :] = scaler.transform(ds_val[i, :, :])
    ds_val_max = ds_val_scale[i, :, :].max()
    ds_val_min = ds_val_scale[i, :, :].min()
    if abs(ds_val_max - ds_val_min) > 10 ** -6:
        ds_val_scale[i, :, :] = (ds_val_scale[i, :, :] - ds_val_min) / \
                                  (max(min_amp, ds_val_max) - ds_val_min)

# # # CNN model
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

input_2 = tf.keras.layers.Input(shape=(22))
input_1 = tf.keras.layers.Input(shape=(512, 128, 1))
model = layers.Conv2D(64, (3, 3), activation='relu', input_shape=(512, 128, 1))(input_1)
mdoel = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Conv2D(32, (3, 3), activation='relu')(model)
model = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Conv2D(32, (3, 3), activation='relu')(model)
model = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Flatten()(model)
model = layers.Dense(64, activation='relu')(model)
model = layers.Dropout(0.01)(model)
concatenated = tf.keras.layers.concatenate([model, input_2])
model_final = tf.keras.layers.Dense(32, activation='softmax')(concatenated)
model_final = layers.Dropout(0.01)(model_final)
out = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(model_final)

model = tf.keras.Model([input_1, input_2], out)

model.summary()

opt = tf.keras.optimizers.experimental.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None,
                                             use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None,
                                             jit_compile=True, name='Nadam')

model.compile(optimizer=opt,
              loss=tf.keras.losses.MSE,
              metrics=['mse'])

history = model.fit([ds_train_scale, add_train], y_train, epochs=4,
                    validation_data=([ds_val_scale, add_val], y_val), batch_size=16)

plt.plot(history.history['mse'], label='mse')
plt.plot(history.history['val_mse'], label='val_mse')
plt.xlabel('Epoch')
plt.ylabel('mse')
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate([ds_test_scale, add_test], y_test, verbose=2)
print(test_acc)

fig = plt.figure(figsize=(10, 10))
plt.scatter(y_train, model.predict([ds_train_scale, add_train]), label='train')
plt.scatter(y_test, model.predict([ds_test_scale, add_test]), label='test')
plt.plot([0,8],[0,8])
plt.xlabel("true")
plt.ylabel("predicted")
plt.legend()
plt.show()

pass