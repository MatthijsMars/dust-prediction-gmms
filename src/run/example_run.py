import numpy as np
from src.models.GMM_sk import *
from astropy.io import fits
import os


## Loading data
x = np.array([ fits.getdata(f"./data/raw/mock{i}.fits") for i in range(5) ]) # The (flattened) maps you use to predict   
y = fits.getdata(f"./data/raw/mock6.fits") # the (flattened) map you want to predict

X =  np.hstack( ( x, y.reshape(-1,1) )) # Combining the maps into 1 dataset

use_log = True # For using logarithmic scaling instead of linear scaling

if use_log:
    m = np.nanmin(X, axis=0)
    X -= 1.001*m * (m < 0) # correcting for negative values if there are any

    X = np.log(X)

# Loading/Creating model
# Pre-trained models can be found in the ./models/ folder. Check the README for how these models can be used. 
model_name = "mock_model"  # Model name
model = gmm(save_name=model_name, whiten=True, n_components=3)

# Fitting the data, if there is already a trained model with the "model_name" it will load this instead
sel = np.isfinite(X).all(axis=1) 
model = model.fit(X[sel]) 

# Predicting
# Output is saved periodically
# Note that the periodically saved files are not corrected for logarithmic scaling
n_jobs = 15 # The amount of cores to use
output_name = "prediction.fits" # Location to save the file in ./data/processed/
prediction = model.pdf_optimum(X, shape=y.shape,  fname=output_name, n_jobs=njobs)

## Saving
# Saving the final prediction and undoing for the pre-proccesing
if use_log:
    prediction = 1.001*m[-1] * (m[-1] < 0) + np.exp( prediction.reshape(y.shape))
else:
    prediction = prediction.reshape(y.shape)

fits.writeto(os.path.join("./data/processed/",  output_name, ".fits"), prediction, overwrite=True)
