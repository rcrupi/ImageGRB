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
list_tte_pkl = [i for i in os.listdir(PATH_GRB_PKL) if 'pickle' in i][0:1000]
#if i[2:11] in df_tot_z['name'].str.slice(3).to_list()]


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

ds_train, lst_train = load_set(list_tte_pkl)

# # # CNN model
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

input_1 = tf.keras.layers.Input(shape=(512, 128, 1))
model = layers.Conv2D(16, (3, 3), activation='relu', input_shape=(512, 128, 1))(input_1)
mdoel = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Conv2D(8, (3, 3), activation='relu')(model)
model = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Conv2D(8, (3, 3), activation='relu')(model)
model = layers.MaxPooling2D((2, 2))(model)
model = layers.Dropout(0.01)(model)
model = layers.Flatten()(model)
model = layers.Dense(16, activation='relu')(model)
model = layers.Dropout(0.01)(model)
model = tf.keras.layers.Dense(2, activation='softmax', name='output_layer')(model)

model = tf.keras.Model(input_1, model)

model.summary()

opt = tf.keras.optimizers.experimental.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None,
                                             use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None,
                                             jit_compile=True, name='Nadam')

model.compile(optimizer=opt,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

y_train = np.random.binomial(n=1, p=0.5, size=ds_train.shape[0])
for i in range(0, 8):
    history = model.fit(ds_train, y_train, epochs=1,
                        validation_split=0.2, batch_size=16)
    # plt.figure()
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')

    y_train = 1*(model.predict(ds_train)[:, 1] >=0.5) # np.random.binomial(n=1, p=model.predict(ds_train)[:, 1])

pass