from model.download_grb_tte import download_tte
from model.preprocess_tte2mat import tte2mat
from model.preprocess_mat2img import mat2img

# # # Download GRB TTE
download_tte(download=True)

# # # Preprocess the data
# Convert TTE to 2D array, background subtracted
tte2mat()
# Convert matrix data into images
ds_train, ds_train_scale, lst_name = mat2img()
