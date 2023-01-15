# Sepsis_CNNect
## M2 GENIOMHE - HE, DF, NG

### Aim:
Build a 1-D CNN to accurately predict sepsis using clinical data.
The 1-D CNN should be able to capture information in the temporality of the different measures for a single patient.

### Dataset:
The data for this study was obtained from two geographically distinct U.S. hospital systems with two different electronic medical record systems: Beth Israel Deaconess Medical Center and Emory University Hospital. These data were collected over the past decade with approval from the appropriate Institutional Review Boards and contained labels for 40,336 patients from the two hospital systems.

The data consists of a combination of hourly vital sign summaries, lab values, and static patient descriptions, including a total of 40 clinical variables: 8 vital sign variables, 26 laboratory variables, and 6 demographic variables (Table 1). Altogether, these data include over 2.5 million hourly time windows and 15 million data points.

### Preprocessing:
First part of the preprocess is to be done in the Preprocess_CNN_ML iPython notebook (most of the useful code is in bash).
This notebook allows concatenation of all patients' data into one file as well as computation of the median for each variable on the whole dataset.
It can also be used to detect which columns to discard based on non-existant data.

Secondly, the preprocessing.py script:
- Replaces NAs by the median value of all measures of a variable in a patient if they exist. Otherwise, NAs are replaced by the median of the whole dataset.
- Normalises continuously quantitative columns.

Thirdly, the `get_data.ipynb`and `get_generators.ipynb` notebooks:
- You need to have a directory containing the normalised patient records (one .csv per patient) in the parent directory of this repo
- They will create .csv files that are used by the model to create the batches 

### CNN model:
- `model_final.ipynb`contains the final 1D-CNN model
- Hyperparameter optimisation is achieved through Hyperband, implemented in the keras-tuner package
- The test results are stored in `/Tuning/sepsis_hyperparam/`
- At the end of the script, the model is loaded and predictions for one batch of the test data set are made

### Results:
- Optimised learning rate is 0.0121
- Result metrics on the training and validation data are loss: 0.5 and accuracy: 91.07%
- Finally, test accuracy is 97.61%