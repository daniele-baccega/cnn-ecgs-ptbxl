# Deep learning project (100Hz)
#
# explainability.py
#
# Authors: Daniele Baccega, Andrea Saglietto
# Topic: Deep Learning applied to ECGs
# Dataset: https://physionet.org/content/ptb-xl/1.0.1/
# Description: The PTB-XL ECG dataset is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length
# where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22).
# The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record.
# The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements.
# To ensure comparability of machine learning algorithms trained on the dataset, we provide recommended splits into training and test sets. 


## Import the libraries
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib.pyplot import cm


## Import model, utils and data generator
from model import get_model, get_model_with_smaller_maxpooling, get_model_with_dropout
from utils import process_raw_data, train_val_test_split, load_raw_data, aggregate_diagnostic, interp1d, make_gradcam_heatmap, save_and_display_gradcam
from datagenerator import dataGenerator


## Obtain the number of classes (an argument)
if len(sys.argv) != 2:
  print("Illegal number of parameters. You need to specify the number of classes to use (2, 5 or 20).")
  exit()

num_classes                 = int(sys.argv[1])

if num_classes not in [2, 5, 20]:
  print("The number of classes to use must be equals to 2, 5 or 20.")
  exit()


#  Initialize some variables
path                        = '../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate               = 100
threshold                   = 100
train_folds                 = [1, 2, 3, 4, 5, 6, 7, 8]
val_fold                    = 9
test_fold                   = 10

#  Prepare the dictionary with the different key-value pairs based on the number of classes selected
#  Prepare also the number of output units based on the number of classes selected (we will use one
#  unit with two classes, five units with five classes and twenty units with twenty classes).
classes_dic                 = {}
output_act_fun              = 'softmax'

if args.num_classes == 5:
  classes_dic               = {"NORM": 0, "HYP": 1, "MI": 2, "STTC": 3, "CD": 4}
  output_act_fun            = 'sigmoid'

#  Subclasses that we didn't consider: RVH, SEHYP, PMI.
if args.num_classes == 20:
  classes_dic               = {"NORM": 0, "STTC": 1, "AMI": 2, "IMI": 3, "LAFB/LPFB": 4, \
                               "IRBBB": 5, "LVH": 6, "CLBBB": 7, "NST_": 8, "ISCA": 9, \
                               "CRBBB": 10, "IVCD": 11, "ISC_": 12, "_AVB": 13, "ISCI": 14, \
                               "WPW": 15, "LAO/LAE": 16, "ILBBB": 17, "RAO/RAE": 18, "LMI": 19}
  output_act_fun            = 'sigmoid'

#  Process and save raw data (or load it)
X, Y, sample_weights        = process_raw_data(data_dir_exists,
                                               num_classes,
                                               classes_dic,
                                               sampling_rate,
                                               path,
                                               threshold,
                                               train_folds,
                                               val_fold,
                                               test_fold)

#  Split data into train, validation and test
#  Recommended 10-fold train-test splits (strat_fold) obtained via stratified
#  sampling while respecting patient assignments, i.e. all records of a
#  particular patient were assigned to the same fold. Records in fold 9 and 10
#  underwent at least one human evaluation and are therefore of a particularly
#  high label quality. We therefore propose to use folds 1-8 as training set,
#  fold 9 as validation set and fold 10 as test set.
X_train, y_train, \
_, y_val, \
X_test, y_test, \
_                           = train_val_test_split(data_dir_exists,
                                                   X,
                                                   Y,
                                                   sample_weights,
                                                   num_classes,
                                                   val_fold,
                                                   test_fold)

print("Train labels:\n", y_train.shape)
print("Validation labels:\n", y_val.shape)
print("Test labels:\n", y_test.shape)

#  Take the means and the stds for each lead considering each ECG inside the training set (for the standardization)
leads                       = X_train.shape[1]
samples                     = X_train.shape[2]
means                       = np.zeros((leads, samples, 1))
variances                   = np.zeros((leads, samples, 1))
stds                        = np.zeros((leads, samples, 1))
first_ecg                   = True

for j, x in zip(range(samples), X_train):
  for i, lead in zip(range(leads), x):
    counter                 = j + 1

    if first_ecg:
      means[i, :, 0]        = lead
      variances[i, :, 0]    = 0

      first_ecg             = False
    else:
      means[i, :, 0]        = means[i, :, 0] + (lead - means[i, :, 0]) / counter
      variances[i, :, 0]    = variances[i, :, 0] + ((counter - 1) / counter) * (lead - means[i, :, 0]) ** 2

stds                        = np.sqrt(variances)

#  Create MultiLabelBinarizer object for the one/many-hot encoding
mlb                         = MultiLabelBinarizer()

#  One-hot encoding
y_test                      = mlb.fit_transform(y_test)

#  Prepare the data
#  Prepare the data
X_train                     = np.array(X_train)
X_val                       = np.array(X_val)
X_test                      = np.array(X_test)

#  Reshape the data
_, _, X_test                = reshape_data(X_train, 
                                           X_val,
                                           X_test)

#  Load the model
model = keras.models.load_model("model_best_val_acc.hdf5")

model.summary()

#  Define the sample weights for the training
sample_weights_test         = np.ones(X_test.shape[0])

#  Define some parameters
time_length = 550
padding = [6, 6]
test_data_dir = 'test_data'

shutil.rmtree(test_data_dir)

#  Explainability
prediction                  = model.predict_generator(generator   = dataGenerator(num_classes,
                                                                                  means,
                                                                                  stds,
                                                                                  sample_weights_test,
                                                                                  X_test,
                                                                                  y_test,
                                                                                  1,
                                                                                  True,
                                                                                  time_length,
                                                                                  padding),
                                                      steps       = X_test.shape[0],
                                                      workers     = 1,
                                                      verbose     = 1)

#test_data = np.load(test_data_dir + '/test_1.npy')
test_data = np.load(test_data_dir + '/test_1701.npy')

heatmap = make_gradcam_heatmap(test_data, model, "C27")

# Display heatmap
plt.matshow(heatmap, cmap=cm.jet)
plt.savefig('heatmap.png')

#https://gist.github.com/abap34/2502f7eecd0c9f5b27b27d22e9e1aaf3
save_and_display_gradcam(test_data[0], heatmap)