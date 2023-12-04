'''
  Deep learning project (100Hz)

  revision.py

  Test the CNN on the Georgia dataset (https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database)
  only for IRBBB and CRBBB classes

  Authors: Daniele Baccega, Andrea Saglietto
  Topic: Deep Learning applied to ECGs
  Dataset: https://physionet.org/content/ptb-xl/1.0.1/
  Description: The PTB-XL ECG dataset is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length
  where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22).
  The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record.
  The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements.
  To ensure comparability of machine learning algorithms trained on the dataset, we provide recommended splits into training and test sets. 
'''

## Import the libraries
import os
import numpy as np
import pandas as pd
import pickle
import physionet_challenge_utility_script as pc
import tensorflow
from tensorflow import keras
from keras import callbacks, models

## Import data generator
from datagenerator import dataGenerator


## Function useful to interpolate the data (during the data augmentation)
def interp1d(datum, new_length):
  length                              = datum.shape[1]
    
  return np.array([np.interp(np.linspace(0, length - 1, num=new_length), np.arange(length), lead) for lead in datum])


## Initialize some variables
path                                  = '../../georgia-12lead-ecg-challenge-database/'
num_classes                           = 20
activation_function                   = 'sigmoid'

_, _, labels, ecg_filenames           = pc.import_key_data(path)

SNOMED_scored                         = pd.read_csv("../../SNOMED_mappings_scored.csv", sep=";")
SNOMED_unscored                       = pd.read_csv("../../SNOMED_mappings_unscored.csv", sep=";")
df_labels                             = pc.make_undefined_class(labels, SNOMED_unscored)

y, snomed_classes                     = pc.onehot_encode(df_labels)

SNOMED_dict                           = dict()
for _, row in SNOMED_scored.iterrows():
  SNOMED_dict[row["SNOMED CT Code"]]  = row["Abbreviation"]

classes_dic                           = {"NORM": 0, "STTC": 1, "AMI": 2, "IMI": 3, "LAFB/LPFB": 4, \
                                         "IRBBB": 5, "LVH": 6, "CLBBB": 7, "NST_": 8, "ISCA": 9, \
                                         "CRBBB": 10, "IVCD": 11, "ISC_": 12, "_AVB": 13, "ISCI": 14, \
                                         "WPW": 15, "LAO/LAE": 16, "ILBBB": 17, "RAO/RAE": 18, "LMI": 19}

y_our_model                           = []
ecg_filenames_processed               = []
for i in range(0, len(ecg_filenames)):
  add = False
  y_our_model_local = np.zeros(20)
  for j in range(0, len(snomed_classes)):
    if y[i][j] == 1:
      y_our_model_local[classes_dic[SNOMED_dict[int(snomed_classes[j])]]] = 1
      add = True

  if add:
    signals, _ = pc.load_challenge_data(ecg_filenames[i])
    if signals.shape[1] == 5000:
      ecg_filenames_processed.append(ecg_filenames[i])
      y_our_model.append(y_our_model_local)

y_our_model = np.array(y_our_model)

data = []
for ecg in ecg_filenames_processed:
  signals, fields             = pc.load_challenge_data(ecg)
  signals                     = interp1d(signals, int(signals.shape[1] / 5))
  signals = signals / 4880  # From 4880/mV to 1uV
  data.append(signals)

data = np.array(data)
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

# Load means and stds
with open('means', 'rb') as means_file:
  means        = pickle.load(means_file)

with open('stds', 'rb') as stds_file:
  stds        = pickle.load(stds_file)

#  Load the model at the last epoch
model = models.load_model('checkpoints/model_last_epoch.h5')

#  Load the best model with respect to the validation accuracy
model.load_weights('checkpoints/model_best_val_acc.h5')

model.summary()

sample_weights_test         = np.ones(data.shape[0])

#  Predict the labels of the data inside the test set and save the predictions
y_pred                      = model.predict(dataGenerator(num_classes,
                                                          activation_function,
                                                          means,
                                                          stds,
                                                          sample_weights_test,
                                                          data,
                                                          y_our_model,
                                                          1),
                                            steps   = data.shape[0],
                                            workers = 1,
                                            verbose = 1)

#  Save the predictions
with open('y_pred_Georgia', 'wb') as y_pred_file:
  pickle.dump(y_pred, y_pred_file)

with open('y_test_Georgia', 'wb') as y_test_file:
  pickle.dump(y_our_model, y_test_file)