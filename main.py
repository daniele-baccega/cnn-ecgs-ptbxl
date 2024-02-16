'''
  Deep learning project (100Hz)

  main.py

  Runs a specific classification task (binary classification, five-class multi-label classification or
  twenty-class multi-label classification).

  Inputs:
     --seed:                     random seed
     --num_classes:              classification task (binary classification, five-class multi-label classification,
                                 twenty-class multi-label classification or hierarchical twenty-four-class multi-label classification)
     --dataset:                  path to the dataset
     --epochs:                   number of epochs
     --batch_size:               batch size
     --init_learning_rate:       initial learning rate
     --final_learning_rate:      final learning rate
     --leads:                    leads to use
     --crop_window:              dimension of the window used for the cropping
     --padding:                  number of zeros to add before and after each cropped lead
     --time_scale:               interval used to alter a bit the frequency of the ECGs
     --amplitude_scale:          interval used to alter a bit the voltage of the ECGs
     --optimizer:                optimizer
     --momentum:                 momentum for SGD optimizer
     --dropout:                  spatial dropout before the last convolutional layer
     --pool_sizes:               size of the six max pooling layers
     --dilation_factors:         dilation factors of the six convolutional 'temporal' layers
     --kernel_size_last_conv:    kernel size of the last convolutional 'spatial' layer
     --out_act_fun_2_classes:    output activation function for the two classese classification task
     --rpeak:                    R-peak pose normalization data augmentation technique
     --different_filters:        use different filters for each lead with the 2D model
     --num_filters_first_conv:   number of filters in the first convolutional layer
     --one_d_model:              use the 1D model instead of the 2D model
     --jitter_std:               interval used to extract a uniform standard deviation for the random jitter data augmentation technique

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
import wfdb
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow import keras
from keras import callbacks
from sklearn.preprocessing import MultiLabelBinarizer


## Import model, utils and data generator
from model import get_model_2D, get_model_2D_different_filters, get_model_1D
from utils import parse_arguments, setup_seed, get_classes_dic_and_output_activation_function, process_raw_data, train_val_test_split, reshape_data, correlation_matrix
from datagenerator import dataGenerator


## Verify what tensorflow version is used
print("Using tensorflow version " + str(tensorflow.__version__))
print("Using keras version " + str(tensorflow.keras.__version__))

#  This prevents tensorflow from allocating all memory on GPU - for TensorFlow 2.2+
gpus                          = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tensorflow.config.experimental.set_memory_growth(gpu, True)


## Parse the arguments
args                          = parse_arguments()
print(args)

if args.num_classes not in [2, 5, 20, 24]:
  print("The number of classes must be equals to 2, 5, 20 or 24.")
  exit()

if args.optimizer not in ["Adam", "SGD", "Adagrad"]:
  print("The optimizer must be equals to Adam, SGD or Adagrad.")
  exit()

if args.out_act_fun_2_classes not in ["sigmoid", "softmax"]:
  print("The output activation function for the two classes classification task must be equals to sigmoid or softmax.")
  exit()


## Check the existence of the backup data directory
data_dir_exists               = os.path.isdir('../../data/' + str(args.num_classes) + 'Classes')

if not data_dir_exists:
  os.mkdir('../../data/' + str(args.num_classes) + 'Classes')


## Set the seeds (for numpy.random and tensorflow.random)
setup_seed(args.seed)


## Visualize the data
#  Initialize some variables
path                          = args.dataset + '/'
sampling_rate                 = 100
resolution                    = "lr" if sampling_rate == 100 else "hr"
likelihood_threshold          = 100
train_folds                   = [1, 2, 3, 4, 5, 6, 7, 8]
val_fold                      = 9
test_fold                     = 10

#  Visualize some data
record                        = wfdb.rdrecord(path + 'records' + str(sampling_rate) + '/00000/00001_' + resolution)
fig                           = wfdb.plot_wfdb(record=record, title='Example signals', figsize = (25, 16), return_fig=True)
fig.savefig("Example1.png")

record                        = wfdb.rdrecord(path + 'records' + str(sampling_rate) + '/00000/00002_' + resolution)
fig                           = wfdb.plot_wfdb(record=record, title='Example signals', figsize = (25, 16), return_fig=True)
fig.savefig("Example2.png")


## Inspect the ptbxl_database.csv
df                            = pd.read_csv(path + 'ptbxl_database.csv',)
print(df)


## Preprocessing
#  Inspect an ECG
signals, fields               = wfdb.rdsamp(path + 'records' + str(sampling_rate) + '/00000/00001_' + resolution)
print(fields)
print("Inspect a .dat file")
print("Signals:\n", signals)
print("Shape:", signals.shape)

print("\n\nInspect a .hea file")
print("Fields:\n", fields)

#  Select the leads
selected_leads_indeces        = [i for i in range(0, len(fields["sig_name"])) if fields["sig_name"][i] in args.leads]
selected_leads_name           = [fields["sig_name"][i] for i in selected_leads_indeces]

#  Establish the output activation function of the model and define the dictionary
#  to associate a number to each class (based on the selected number of classes).
classes_dic_5classes, \
classes_dic_20classes, \
classes_dic                   = get_classes_dic_and_output_activation_function(args.num_classes)

#  Process and save raw data (or load it)
X, Y, sample_weights          = process_raw_data(data_dir_exists,
                                                 args.num_classes,
                                                 classes_dic_5classes,
                                                 classes_dic_20classes,
                                                 classes_dic,
                                                 sampling_rate,
                                                 path,
                                                 selected_leads_indeces,
                                                 likelihood_threshold,
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
X_val, y_val, \
X_test, y_test, \
sample_weights_train          = train_val_test_split(data_dir_exists,
                                                     X,
                                                     Y,
                                                     sample_weights,
                                                     args.num_classes,
                                                     val_fold,
                                                     test_fold)

del X, Y, sample_weights

print("Train labels:\n", y_train)
print("Validation labels:\n", y_val)
print("Test labels:\n", y_test)

#  Take the means and the stds for each lead considering each ECG inside the training set (for the standardization)
leads                         = X_train.shape[1]
if not os.path.exists('../means') or not os.path.exists('../stds'):
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

  # Log means and stds
  with open('../means', 'wb') as means_file:
    pickle.dump(means, means_file)

  with open('../stds', 'wb') as stds_file:
    pickle.dump(stds, stds_file)
else:
  # Load means and stds
  with open('../means', 'rb') as means_file:
    means                     = pickle.load(means_file)

  with open('../stds', 'rb') as stds_file:
    stds                      = pickle.load(stds_file)

#  Create MultiLabelBinarizer object for the one/many-hot encoding
mlb                           = MultiLabelBinarizer()

#  One-hot encoding
y_train                       = mlb.fit_transform(y_train)
y_val                         = mlb.fit_transform(y_val)
y_test                        = mlb.fit_transform(y_test)

if args.num_classes == 2 and args.out_act_fun_2_classes == "sigmoid":
  y_train                     = np.argmax(y_train, axis=1)
  y_val                       = np.argmax(y_val, axis=1)
  y_test                      = np.argmax(y_test, axis=1)

print("Train labels:\n", y_train)
print("Validation labels:\n", y_val)
print("Test labels:\n", y_test)

#  Prepare the data
X_train                       = np.array(X_train)
X_val                         = np.array(X_val)
X_test                        = np.array(X_test)

#  Reshape the data
X_train, X_val, X_test        = reshape_data(X_train, 
                                             X_val,
                                             X_test)

print("Train data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)
print("Train labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)
print("Test labels shape:", y_test.shape)

#  Compute the average correlation between the leads
correlation_matrix(np.concatenate(((X_train - means) / stds, (X_val - means) / stds, (X_test - means) / stds)), leads, selected_leads_name)

#  Define the sample weights for the training, validation and test set
sample_weights_train          = np.ones(X_train.shape[0])
sample_weights_val            = np.ones(X_val.shape[0])
sample_weights_test           = np.ones(X_test.shape[0])


## Build the CNN
#  Choose and generate the model
model_function                = get_model_2D

if args.different_filters:
  model                       = get_model_2D_different_filters

if args.one_d_model:
  model                       = get_model_1D

model                         = model_function(args.init_learning_rate,
                                               leads,
                                               args.num_classes,
                                               args.crop_window + args.padding,
                                               args.optimizer,
                                               args.momentum,
                                               args.dropout,
                                               args.pool_sizes,
                                               args.dilation_factors,
                                               args.kernel_size_last_conv,
                                               args.out_act_fun_2_classes,
                                               args.num_filters_first_conv)

#  Prepare the data necessary to use a decayed learning rate
n_steps_per_epoch             = X_train.shape[0] // args.batch_size

lr_boundaries                 = [x for x in np.linspace(1, args.epochs-1, args.epochs - 1, dtype=int)]
lr_values                     = [x for x in np.linspace(args.init_learning_rate, args.final_learning_rate, args.epochs)]

#  Function useful to implement the decaying of the learning rate
def learning_rate_decay(epoch):
  lr                          = None
  if epoch <= lr_boundaries[0]:
    lr                        = lr_values[0]

  if epoch > lr_boundaries[-1]:
    lr                        = lr_values[-1]

  for i in range(1, len(lr_boundaries)):
    if epoch > lr_boundaries[i-1] and epoch <= lr_boundaries[i]:
      lr                      = lr_values[i]

  return lr

#  Callback useful to save the model after each epoch and to print the learning rate
class ModelSave(keras.callbacks.Callback):
  def __init__(self) -> None:
        super().__init__()

  def on_epoch_begin(self, batch, logs=None):
    lr                        = "LR - {}\n".format((tensorflow.keras.backend.get_value(
                                  self.model.optimizer.lr
                                )))
    
    with open("lr.txt", "a+") as f:
      f.write(lr)
      
    print(lr)

    return lr

  def on_epoch_end(self, epoch, logs=None):
    self.model.save("checkpoints/model_{}.h5".format(epoch))

#  Callback useful to evaluate the model also on the test set after each epoch
class AdditionalValidationSets(keras.callbacks.Callback):
  def __init__(self, validation_sets):
    super(AdditionalValidationSets, self).__init__()
    self.validation_sets      = validation_sets

    for validation_set in self.validation_sets:
      if len(validation_set) not in [3, 4]:
        raise ValueError()

    self.epoch                = []
    self.history              = {}

  def on_train_begin(self, logs=None):
    self.epoch                = []
    self.history              = {}

  def on_epoch_end(self, epoch, logs=None):
    logs                      = logs or {}
    self.epoch.append(epoch)

    # Record the same values as History() as well
    for k, v in logs.items():
      self.history.setdefault(k, []).append(v)

    # Evaluate on the additional validation sets
    for validation_set in self.validation_sets:
      if len(validation_set) == 3:
        validation_data, \
        validation_targets, \
        validation_set_name   = validation_set
        sample_weights        = None
      elif len(validation_set) == 4:
        validation_data, \
        validation_targets, \
        sample_weights, \
        validation_set_name   = validation_set
      else:
        raise ValueError()

      results                 = self.model.evaluate(dataGenerator(sampling_rate,
                                                                  args.num_classes,
                                                                  args.out_act_fun_2_classes,
                                                                  means,
                                                                  stds,
                                                                  sample_weights_val,
                                                                  validation_data,
                                                                  validation_targets,
                                                                  args.batch_size,
                                                                  args.one_d_model,
                                                                  False,
                                                                  args.different_filters,
                                                                  False,
                                                                  args.crop_window,
                                                                  args.padding),
                                                    steps   = validation_data.shape[0] // args.batch_size,
                                                    workers = 1,
                                                    verbose = 1)

      for metric, result in zip(self.model.metrics_names, results):
        valuename             = validation_set_name + '_' + metric
        self.history.setdefault(valuename, []).append(result)

checkpoint_val_accuracy       = callbacks.ModelCheckpoint('checkpoints/model_best_val_acc.h5', monitor='val_binary_accuracy', save_weights_only=True, save_best_only=True, verbose=1, mode='max')
csv_logger                    = callbacks.CSVLogger(os.path.join('train.log'), append=True, separator=';')
history_test_as_val           = AdditionalValidationSets([(X_test, y_test, 'test')])
lr_decay                      = callbacks.LearningRateScheduler(learning_rate_decay)
model_save                    = ModelSave()

#  Train the model
history                       = model.fit(dataGenerator(sampling_rate,
                                                        args.num_classes,
                                                        args.out_act_fun_2_classes,
                                                        means,
                                                        stds,
                                                        sample_weights_train,
                                                        X_train,
                                                        y_train,
                                                        args.batch_size,
                                                        args.one_d_model,
                                                        args.rpeak,
                                                        args.different_filters,
                                                        False,
                                                        args.crop_window,
                                                        args.padding,
                                                        args.jitter_std,
                                                        args.amplitude_scale,
                                                        args.time_scale),
                                          steps_per_epoch   = X_train.shape[0] // args.batch_size,
                                          epochs            = args.epochs,
                                          validation_data   = dataGenerator(sampling_rate,
                                                                            args.num_classes,
                                                                            args.out_act_fun_2_classes,
                                                                            means,
                                                                            stds,
                                                                            sample_weights_val,
                                                                            X_val,
                                                                            y_val,
                                                                            args.batch_size,
                                                                            args.one_d_model,
                                                                            False,
                                                                            args.different_filters,
                                                                            False,
                                                                            args.crop_window,
                                                                            args.padding),
                                          validation_steps  = X_val.shape[0] // args.batch_size,
                                          callbacks         = [lr_decay, model_save, checkpoint_val_accuracy, csv_logger, history_test_as_val],
                                          shuffle           = True,
                                          workers           = 1,
                                          verbose           = 1)

#  Save the training history
with open('history', 'wb') as file_pi:
  pickle.dump(history_test_as_val.history, file_pi)

#  Save the model at the last epoch
model.save("checkpoints/model_last_epoch.h5")

#  Load the best model with respect to the validation accuracy
model.load_weights("checkpoints/model_best_val_acc.h5")

#  Predict the labels of the data inside the test set and save the predictions
y_pred                        = model.predict(dataGenerator(sampling_rate,
                                                            args.num_classes,
                                                            args.out_act_fun_2_classes,
                                                            means,
                                                            stds,
                                                            sample_weights_test,
                                                            X_test,
                                                            y_test,
                                                            1,
                                                            args.one_d_model,
                                                            False,
                                                            args.different_filters,
                                                            False,
                                                            args.crop_window,
                                                            args.padding),
                                              steps   = X_test.shape[0],
                                              workers = 1,
                                              verbose = 1)

#  Save the predictions
with open('y_pred', 'wb') as y_pred_file:
  pickle.dump(y_pred, y_pred_file)

with open('y_test', 'wb') as y_test_file:
  pickle.dump(y_test, y_test_file)