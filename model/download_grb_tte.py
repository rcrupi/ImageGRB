# stadard packages
import os
import pandas as pd
import numpy as np
# gbm data tools import
from gbm.finder import BurstCatalog, TriggerFtp
# local import
from config.config import PATH_GRB_TTE, GBM_BURST_DB


def download_tte(download=True, k=1, bool_overwrite=False):
    """
    Download 14 TTE GRBs, one per each detectors, in the folder PATH_GRB_TTE.
    :param download: If False return instantly None
    :param k: numeber of tries to download a GRB
    :param bool_overwrite: If True overwrite csv dataframe
    :return: None
    """
    if not download:
        return None

    print("Download GRB catalog metadata.")
    burstcat = BurstCatalog()
    df_burst = pd.DataFrame(burstcat.get_table())
    df_burst.to_csv(GBM_BURST_DB, index=False)

    # Run k times the download to be sure that a day is downloaded
    for round in range(0, k):
        # List csv files
        list_csv = [i for i in os.listdir(PATH_GRB_TTE)]
        # Cycle for each Burst day
        for row in df_burst['trigger_name']:
            print("Downloading GRB: ", row)
            try:
                # Check if dataset csv is already computed
                if np.sum([row in i for i in list_csv]) < 14 and not bool_overwrite:
                    # Define what day download
                    print('Initialise connection FTP for trigger: ', row)
                    trig_find = TriggerFtp(row[2:])
                    # download tte. For ctime use .get_ctime and for cspec use get_cspec
                    trig_find.get_tte(PATH_GRB_TTE)
            except Exception as e:
                print(e)
                print('Error for file: ' + row[2:])
    return None


if __name__ == "__main__":
    download_tte()
