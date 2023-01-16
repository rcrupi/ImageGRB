# Standard package
import os
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
# GBM data tools import bkg fitter
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
# Preprocess data downloaded from FTP
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
# local import
from config.config import PATH_GRB_TTE, PATH_GRB_PKL, GBM_BURST_DB
sns.set_theme()


def det_triggered(str_mask):
    """
    Method that return only the triggered detectors for a GRB.
    :param str_mask: 12 mask binary values one per each detector. E.g. 101000010000.
    :return: list of triggered detectors
    """
    list_det = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                         'n8', 'n9', 'na', 'nb'])
    try:
        idx_det = np.where(np.array([int(i) for i in list(str_mask)]) == 1)
        # Do not consider BGO detectors
        idx_det = np.array([idx_tmp for idx_tmp in idx_det[0] if idx_tmp < 12])
        return list(list_det[idx_det])
    except:
        print("Warning, not found detectors triggered. det_mask: ", str_mask)
        return list(list_det)


def tte2mat(LEN_LC=512, TIME_RES=0.064, bln_run=True):
    """
    Subtract bakground from tte and save 2D rate counts as pickles.
    :param LEN_LC: Horizontal (temporal) length of the image. E.g. 512 or 64.
    :param TIME_RES: Maximum time resolution allowed
    :param bln_run: If False the run is stopped
    :return: None
    """
    if not bln_run:
        return None

    # List of all TTE
    list_tte = os.listdir(PATH_GRB_TTE)
    # List of all GRB background subtracted and saved as a 2D array
    list_tte_pkl = [i for i in os.listdir(PATH_GRB_PKL) if 'pickle' in i]
    list_tte_bn_pkl = ["_".join(i.split('_')[0:2]) for i in list_tte_pkl]

    # Load locally the GBM burst in ImageGRB/data to faster load
    df_burst = pd.read_csv(GBM_BURST_DB)

    # Initialise tte cycle
    np.random.seed(42)
    for tte_tmp in list_tte:
        # If TTE already processed continue to the next one
        if tte_tmp.split('_')[3] + "_" + tte_tmp.split('_')[2] in list_tte_bn_pkl:
            continue
        # Problems in loop: glg_tte_n3_bn090626707_v00, glg_tte_n8_bn090626707_v00, glg_tte_na_bn090626707_v00, glg_tte_n0_bn090626707_v00,
        # glg_tte_nb_bn090626707_v00, glg_tte_n6_bn090626707_v00, glg_tte_n4_bn090626707_v00, glg_tte_n2_bn090626707_v00,
        # glg_tte_n5_bn090626707_v00, glg_tte_n9_bn090626707_v00, glg_tte_n1_bn090626707_v00, glg_tte_n7_bn090626707_v00
        # Problem in fit: glg_tte_n0_bn121029350_v00, glg_tte_n2_bn120922939_v00, glg_tte_n0_bn121125469_v00,
        # glg_tte_n5_bn121029350_v00, glg_tte_n9_bn191130507_v00, glg_tte_n7_bn120415891_v00, glg_tte_n0_bn121125356_v00
        try:
            # Check if detector has event signal counts
            str_det = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 'bcat_detector_mask'].values[0]
            list_det = det_triggered(str_mask=str_det)
            t90_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90'].values[0]
            t90_start_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90_start'].values[0]
            t50_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't50'].values[0]
            t90_e_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90_error'].values[0]
            # If detector is not triggered go to the next file
            if tte_tmp.split('_')[2] not in list_det:
                continue
            # read a tte file
            tte = TTE.open(PATH_GRB_TTE + tte_tmp)
            print(tte)
            t_min, t_max = tte.time_range
            # bkg times
            # giovanni
            # [-15, -sigma], [t90+sigma, t90+75]
            # mio
            # GRB191130253 - set interval boundaries not with only t90_error (-15, +30)
            try:
                bkg_t_min_2 = max(t90_start_tmp - 3 * t90_e_tmp, t_min + 15)
                bkg_t_min_1 = max(bkg_t_min_2 - 15, t_min)
                bkg_t_max_1 = min(t90_start_tmp + t90_tmp + 3 * t90_e_tmp, t_max - 30)
                bkg_t_max_2 = min(t_max, bkg_t_max_1 + 30)
            except:
                print("Error: Background times ruins.")
                bkg_t_min_2 = -1
                bkg_t_min_1 = -2
                bkg_t_max_1 = 1
                bkg_t_max_2 = 2

            # Selection interval data
            bkg_t_sel_1 = max(t_min, bkg_t_min_2 - (bkg_t_max_1 - bkg_t_min_2) / 2)
            bkg_t_sel_2 = min(t_max, bkg_t_max_1 + (bkg_t_max_1 - bkg_t_min_2) / 2)

            bkgd_times = [(bkg_t_min_1, bkg_t_min_2), (bkg_t_max_1, bkg_t_max_2)]
            # bin in time (0.064s)
            flt_bin_time = max((bkg_t_sel_2 - bkg_t_sel_1) / (LEN_LC - 1),
                               TIME_RES / 64)  # max(t50_tmp/LEN_LC, TIME_RES)
            # Bin the TTE in time by flt_bin_time
            phaii = tte.to_phaii(bin_by_time, flt_bin_time, time_ref=0.0)
            type(phaii)
            # If the bin time is too small (less than TIME_RES) background estimation is not performed.
            # This is equivalent to require (bkg_t_sel_2 - bkg_t_sel_1) >= (LEN_LC-1)*TIME_RES
            if flt_bin_time >= TIME_RES:
                # Initialize polynomial background fitter with the phaii object and the time ranges to fit.
                backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
                try:
                    # Fit with 1st order polynomial
                    backfitter.fit(order=1)
                except Exception as e:
                    print(e)
                    print("Errore fit", tte_tmp.split('_')[3], t90_tmp, bkgd_times)
                    continue
                # Interpolate the bkg
                bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
                type(bkgd)

                # Select lighturve (count rates per each 128 energy channel) in the selected background interval
                pdc = phaii.data.rates - bkgd.rates
                pdc = pdc[(phaii.data.time_centroids >= bkg_t_sel_1 - flt_bin_time / 2) &
                          (phaii.data.time_centroids <= bkg_t_sel_2 + flt_bin_time / 2)]
                print(pdc.shape)
                # plt.plot(pdc.sum(axis=1))
                # # plot the lightcurve
                # lcplot = Lightcurve(data=phaii.to_lightcurve(time_range=(bkg_t_sel_1, bkg_t_sel_2), energy_range=(8, 900)),
                #                     background=bkgd.integrate_energy(emin=8, emax=900))
                # lcplot.add_selection(phaii.to_lightcurve(time_range=(bkg_t_min_2, bkg_t_max_1), energy_range=(8, 900)))
                # plt.show()
            else:
                # The time interval is too small for computing the Polynomial bkg. Only are removed.
                # pdc = phaii.data.rates - np.quantile(phaii.data.rates, q=0.5, axis=0, keepdims=True, interpolation='linear')
                # TODO make more robust the mean. Bkg order 0?
                pdc = phaii.data.rates - np.mean(phaii.data.rates, axis=0, keepdims=True)
                pdc = pdc[(phaii.data.time_centroids >= bkg_t_sel_1 - flt_bin_time / 2) &
                          (phaii.data.time_centroids <= bkg_t_sel_2 + flt_bin_time / 2)]
                print(pdc.shape)
                # plt.plot(pdc.sum(axis=1))
                # # plot the lightcurve
                # lcplot = Lightcurve(data=phaii.to_lightcurve(time_range=(bkg_t_sel_1, bkg_t_sel_2), energy_range=(8, 900)))
                # lcplot.add_selection(phaii.to_lightcurve(time_range=(bkg_t_min_2, bkg_t_max_1), energy_range=(8, 900)))
            # plt.show()
            # continue

            # Pad the values up to LEN_LC
            if pdc.shape[0] > LEN_LC:
                print("Warning. An event is cut for the sake of dimensions. dim original: " + str(pdc.shape[0]))
                pdc = pdc[0:LEN_LC, :]
            elif LEN_LC > pdc.shape[0]:
                if pdc.shape[0] == 0:
                    print("Error. No data in ", tte_tmp.split('_')[3], t90_tmp)
                # Padding with 0
                # pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='edge')
                diff_len = LEN_LC - pdc.shape[0]
                pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='constant', constant_values=0)
                print("Logging. Padded series.")

            # Save GRB matrix into pickle
            name_file = str(tte_tmp.split('_')[3]) + "_" + str(tte_tmp.split('_')[2]) + "_bin" + str(
                round(flt_bin_time, 4))
            with open(PATH_GRB_PKL + name_file + '.pickle', 'wb') as f:
                pickle.dump(pdc, f)
        except Exception as e:
            print(e)
            print("Error in loop.", tte_tmp.split('_')[3])
    return None


if __name__ == '__main__':
    tte2mat()
