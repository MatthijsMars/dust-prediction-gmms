<h2>Predicting Interstellar dust using Gaussian misture modelling


This code is used to predict interstellar dust structures using data from Planck, WISE and CFHT. 

The GMM is an adaptation of the Scikit-learn model and can be found in "./src/models/GMM_sk.py". 

An example script of using the model can be found in "./src/run/example_run.py"

The requirements for running the scripts can be found in the "./requirements.txt" file

The steps to using this code are:

1. Load in the data:
The data consists of data you want to use to predict the dust structures (x) and data of the dust you want to predict (y). If you're using one of the pre-trained models the latter is not necessary and can be replaced by an empty array in the code. Make sure the dimensions of all the map are the same. 

2. Think of if you want to use Linear or Logarithmic scaling:
For features where there is a large difference between the low and high intensity pixels (~100x difference) you can use a Logarithmic scaling to make sure the model also learns the faint features and is less biased towards the high intensity features.

3. Load or fit the model:
You can use one of the pre-trained models found in "./models/". Make sure that the pre-trained models use the same data and scaling as in training. The available models are:

- "planck_n3_full_sky" : A model trained on the full planck sky used to predict the Planck 857 GHz data. Input data: Planck 100 GHz, 143 GHz, 217 GHz, 353 GHz,
and 545 GHz data (can be found at: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/). Uses logarithmic scaling.

- "planck_n3_partial_sky" : The same model as "planck_n3_full_sky", yet trained on only ~1% of the total sky.  Input data: Planck 100 GHz, 143 GHz, 217 GHz, 353 GHz,
and 545 GHz data (can be found at: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/). Uses logarithmic scaling.

- "wise_n3_w212" : A model trained to predict WISE data using data from the Planck mission.  Input data: Planck 100 GHz, 143 GHz, 217 GHz, 353 GHz, 545, and 857 GHz data (can be found at: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/). Uses logarithmic scaling.

- "cfht_model" : A model trained to predict dust in optical observations using data from Planck and WISE: Planck 100 GHz, 143 GHz, 217 GHz, 353 GHz, 545, and 857 GHz data (can be found at: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/) and WISE data (can be found at: wise.skymaps.info). Uses linear scaling. Make sure that you project the data from Planck to the same grid as the WISE data. The best way to do this is by using the "reproject_from_healpix" function in the "reproject" package (https://reproject.readthedocs.io/en/stable/api/reproject.reproject_from_healpix.html#reproject.reproject_from_healpix)