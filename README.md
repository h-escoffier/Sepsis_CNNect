# Sepsis_CNNect
## M2 GENIOMHE - HE, DF, NG

### Aim:
Build a 1-D CNN to accurately predict sepsis using clinical data.
The 1-D CNN should be able to capture information in the temporality of the different measures for a single patient.

### Dataset:


### Preprocessing:
First part of the preprocess is to be done in the Preprocess_CNN_ML iPython notebook (most of the useful code is in bash).
This notebook allows concatenation of all patients' data into one file as well as computation of the median for each variable on the whole dataset.
It can also be used to detect which columns to discard based on non-existant data.

Secondly, the preprocessing.py script:
- Replaces NAs by the median value of all measures of a variable in a patient if they exist. Otherwise, NAs are replaced by the median of the whole dataset.
- Normalises continuously quantitative columns.

### CNN model:

### Results:

### Discussion:

### References:
