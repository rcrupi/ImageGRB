# ImageGRB
Convert a GRB event in a pretty image.

## Installation
Create a new venv, install the library specified in the requirements.txt and finally install GBM data tools

```bash
python3 -m venv gbm
source gbm/bin/activate
cd <PATH_OF_THE_REPO>
pip install -r requirements.txt
wget https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gbm_data_tools-1.1.1.tar.gz
pip install gbm_data_tools-1.1.1.tar.gz
```

## Config
In the config folder you must create a file named 'path.yaml'.
``` yaml
# Example YAML
PATH_TO_SAVE: /.../<PATH_FOLDER_WITH_A_LOT_OF_SPACE>/
GBM_BURST_DB: data/gbm_burst_catalog.csv
GRB_REDSHIFT: data/grb_red.csv
GRB_TTE: tte/
GRB_PKL: tte_pkl_img/
GRB_IMG: tte_img_scaled/
```

## Usage
In the folder 'script' there is the pipeline.py:

```python
# # # Download GRB TTE
download_tte(download=True)

# # # Preprocess the data
# Convert TTE to 2D array, background subtracted
tte2mat()
# Convert matrix data into images
ds_train, ds_train_scale, lst_name = mat2img()
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
