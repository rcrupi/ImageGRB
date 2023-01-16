# Standard package
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
# from PIL import Image
import pandas as pd
import pickle
from tqdm import tqdm
# local import
from config.config import PATH_GRB_PKL, PATH_GRB_IMG


def mat2img(LEN_LC=512, n_load=None, bln_save_img_scale=True, bln_augment=False):
    """
    Method to build the dataset (from PATH_GRB_PKL) and convert 2D array into RGB images (saved in PATH_GRB_IMG).
    :param LEN_LC: length of the image (time length).
    :param n_load: number of event to load. If None all events are loaded.
    :param bln_save_img_scale: If True the RGB image are saved.
    :param bln_augment: If True an event is shifted right and left randomly.
    :return: dataset, dataset scaled, list of GRB names
    """
    ds_train = []
    lst_tte_pkl_aug = []
    np.random.seed(10)

    counter_constant = 0
    # List of all GRB background subtracted and saved as a 2D array
    list_tte_pkl = [i for i in os.listdir(PATH_GRB_PKL) if 'pickle' in i]
    if n_load is not None:
        list_tte_pkl = list_tte_pkl[0:n_load]

    for i in tqdm(list_tte_pkl):
        with open(PATH_GRB_PKL + i, 'rb') as f:
            event_tmp = pickle.load(f)
            if abs(event_tmp.max() - event_tmp.min()) < 10 ** -6:
                counter_constant += 1
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
    print("Event that are constant: ", counter_constant)

    # Scale dataset
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    ds_train_scale = ds_train.copy()
    for i in tqdm(range(0, ds_train.shape[0])):
        ds_train_scale[i, :, :] = StandardScaler().fit_transform(ds_train[i, :, :])
        # ds_train_scale[i, :, :] = MinMaxScaler().fit_transform(ds_train[i, :, :])
        if abs(ds_train_scale[i, :, :].max() - ds_train_scale[i, :, :].min()) > 10 ** -6:
            ds_train_scale[i, :, :] = (ds_train_scale[i, :, :] - ds_train_scale[i, :, :].min()) / \
                                      (ds_train_scale[i, :, :].max() - ds_train_scale[i, :, :].min())
            if bln_save_img_scale:
                path_tmp = PATH_GRB_IMG + lst_tte_pkl_aug[i].replace('.', '') + '.png'
                path_tmp = path_tmp.replace(" ", "_")
                matplotlib.image.imsave(path_tmp, ds_train_scale[i, :, :].T, format='png')
                # # Convert matrix to image matrix
                # from PIL import Image
                # aaa = Image.fromarray(ds_train_scale[i, :, :].T, "RGB")
                # np.asarray(aaa)
        else:
            print("Warning. Signal constant.")
    return ds_train, ds_train_scale, lst_tte_pkl_aug
    # with open("/home/rcrupi/Downloads/grb_train.npy", 'wb') as f:
    #     np.save(f, ds_train_scale)
    # with open("/home/rcrupi/Downloads/grb_train.npy", 'rb') as f:
    #     a = np.load(f)


if __name__ == "__main__":
    # mat2img run
    ds_train, ds_train_scale, _ = mat2img(n_load=10)
    # Plot figure 0
    idx_fig = 0
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(ds_train[idx_fig, :, :].T)
    ax1.set_title('Spectrogram')
    ax1.axis('off')
    ax2.plot(ds_train[idx_fig, :, :].sum(axis=1))
    ax2.set_title('Lightcurve')

    f2, (ax3, ax4) = plt.subplots(2, 1)
    ax3.imshow(ds_train_scale[idx_fig, :, :].T)
    ax3.set_title('Spectrogram scaled')
    ax3.axis('off')
    ax4.plot(ds_train_scale[idx_fig, :, :].sum(axis=1))
    ax4.set_title('Lightcurve scaled')

    print('end')
    pass