import os
import yaml
from pathlib import Path

config_path = os.path.dirname(__file__)
with open(config_path + '/path.yaml') as file:
    path_name = yaml.load(file, Loader=yaml.FullLoader)

local_data_path = os.path.dirname(__file__)

PATH_TO_SAVE = path_name['PATH_TO_SAVE']
GRB_REDSHIFT = Path(local_data_path).parent / path_name['GRB_REDSHIFT']
GBM_BURST_DB = Path(local_data_path).parent / path_name['GBM_BURST_DB']
PATH_GRB_TTE = PATH_TO_SAVE+path_name['GRB_TTE']
# PATH_GRB_CTIME = PATH_TO_SAVE+path_name['GRB_CTIME']
# PATH_GRB_CSPEC = PATH_TO_SAVE+path_name['GRB_CSPEC']
PATH_GRB_PKL = PATH_TO_SAVE+path_name['GRB_PKL']
PATH_GRB_IMG = PATH_TO_SAVE+path_name['GRB_IMG']

Path(PATH_TO_SAVE).mkdir(parents=True, exist_ok=True)
Path(PATH_GRB_TTE).mkdir(parents=True, exist_ok=True)
Path(PATH_GRB_PKL).mkdir(parents=True, exist_ok=True)
Path(PATH_GRB_IMG).mkdir(parents=True, exist_ok=True)
