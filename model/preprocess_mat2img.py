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


def LineStdScaler(img, **kwargs):
    """
    Method to scale the image line by line according to fluctuations outside the GRB.
    :param by_window: select the scale algorithm. When True find minimum STD in time windows.
    :param window: number of samples in the time window (window algorithm).
    :param min_std: minimum meaningful standard deviation (window algorithm).
    :param min_drops: threshold on dropped samples (GRB removal algorithm).
    :param clip: if True clip values below the estimated baseline mean.
    :return: scaled image.
    """
    by_window = kwargs.get('by_window', True)
    window = kwargs.get('window', 32)
    min_std = kwargs.get('min_std', 0.1)
    min_drops = kwargs.get('min_drops', 5)
    clip = kwargs.get('clip', False)
    s = img.shape
    n = s[0]//window
    w = n*window
    drops = lambda n: int(round(np.sqrt(n))) if min_drops is None else min_drops
    # Loop on the image lines
    for i in range(s[1]):
        data = img[:, i]
        # Calculate the standard deviation outside the GRB
        if by_window:   # window with smallest meaningful standard deviation
            std = data[:w].reshape((n, window)).std(axis=1, ddof=1)
            std[std < min_std] = max(1., std.max())
            scale = std.min()
        else:           # loop removing the GRB
            n_old = 2*s[0]
            n = s[0]
            mask = np.ones_like(data, dtype=bool)
            while n > 16 and n_old-n > drops(n_old):
                std = data[mask].std(ddof=1)
                mean = data[mask].mean()
                thr = mean+3.0*std
                mask = data < thr
                n_old, n = n, mask.sum()
            scale = std
        # Scale the image line by its standard deviation
        data /= scale
        # Find the baseline (assuming a constant residual background)
        thr = data.min()+6.0
        mean = data[data < thr].mean()
        # Remove the baseline
        data -= mean
    if clip:
        return np.clip(img, 0, None)
    else:
        return img


def mat2img(LEN_LC=512, n_load=None, bln_save_img_scale=True, bln_augment=False, scaler='Standard', cmap='Greys_r', min_amp=5.0, norm=False, **kwargs):
    """
    Method to build the dataset (from PATH_GRB_PKL) and convert 2D array into RGB images (saved in PATH_GRB_IMG).
    :param LEN_LC: length of the image (time length).
    :param n_load: number of event to load. If None all events are loaded.
    :param bln_save_img_scale: If True the RGB image are saved.
    :param bln_augment: If True an event is shifted right and left randomly.
    :param scaler: scaler function to use ('MinMax', 'Standard', 'LineStd').
    :param cmap: color map.
    :param min_amp: minimum amplitude for image normalization.
    :param norm: boolean uset do activate normalization by to_rgba().
    :param kwargs: dictionary with parameters for LineStdScaler() or other user defined scalers.
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

    # mapper from 2d array to rgba image
    sm = matplotlib.cm.ScalarMappable(cmap)

    # Scale dataset
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    ds_train_scale = ds_train.copy()
    for i in tqdm(range(0, ds_train.shape[0])):
        if scaler == 'MinMax':
            ds_train_scale[i, :, :] = MinMaxScaler().fit_transform(ds_train[i, :, :])
        elif scaler == 'Standard':
            ds_train_scale[i, :, :] = StandardScaler().fit_transform(ds_train[i, :, :])
        else:
            ds_train_scale[i, :, :] = LineStdScaler(ds_train_scale[i, :, :], **kwargs)
        ds_train_max = ds_train_scale[i, :, :].max()
        ds_train_min = ds_train_scale[i, :, :].min()
        if abs(ds_train_max - ds_train_min) > 10 ** -6:
            ds_train_scale[i, :, :] = (ds_train_scale[i, :, :] - ds_train_min) / \
                                      (max(min_amp, ds_train_max) - ds_train_min)
            if bln_save_img_scale:
                rgba = sm.to_rgba(ds_train_scale[i, :, :].T, bytes=True, norm=norm)
                path_tmp = PATH_GRB_IMG + lst_tte_pkl_aug[i].replace('.', '') + '.png'
                path_tmp = path_tmp.replace(" ", "_")
                matplotlib.image.imsave(path_tmp, rgba, format='png')
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
