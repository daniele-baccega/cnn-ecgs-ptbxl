'''
  Deep learning project (100Hz)

  utils.py

  Some useful functions.

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
import ast
import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras.backend as K
from tensorflow import keras


## Utility functions (training)
#  Function useful to parse all the command line arguments
def parse_arguments():
  parser                                = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
  parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (default: 2)')
  parser.add_argument('--dataset', type=str, default='ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', help='Main direcory of the dataset (default: ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1)')
  parser.add_argument('--epochs', type=int, default=200, help='Epochs to train (default: 200)')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
  parser.add_argument('--init_learning_rate', type=float, default=1e-2, help='Initial learning rate (default: 1e-2)')
  parser.add_argument('--final_learning_rate', type=float, default=1e-4, help='Final learning rate (default: 1e-4)')
  parser.add_argument('--leads', '--names-list', nargs='+', default=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], help='Leads (default: [I, II, V1, V2, V3, V4, V5, V6])')
  parser.add_argument('--crop_window', type=int, default=344, help='Dimension of the window used for the cropping; must be greater or equal than the receptive field of the network (default: 344)')
  parser.add_argument('--padding', type=int, default=0, help='Number of zeros to add before and after each cropped lead (default: 0)')
  parser.add_argument('--time_scale', type=float, nargs='+', default=[0.8, 1.2], help='Interval used to alter a bit the frequency of the ECGs (default: [0.8, 1.2])')
  parser.add_argument('--amplitude_scale', type=float, nargs='+', default=[0.7, 1.3], help='Interval used to alter a bit the voltage of the ECGs (default: [0.7, 1.3])')
  parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer (default: Adam)')
  parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)')
  parser.add_argument('--dropout', type=float, default=0, help='Spatial dropout before the last convolutional layer (default: 0)')
  parser.add_argument('--pool_sizes', type=int, nargs='+', default=[2, 2, 2, 2, 2, 2], help='Size of the six max pooling layers (default: [2, 2, 2, 2, 2, 2])')
  parser.add_argument('--dilation_factors', type=int, nargs='+', default=[2, 2, 2, 2, 2, 2], help='Dilation factors of the six convolutional \'temporal\' layers (default: [2, 2, 2, 2, 2, 2])')
  parser.add_argument('--kernel_size_last_conv', type=int, default=1, help='Kernel size of the last convolutional \'spatial\' layer (default: 1)')
  parser.add_argument('--out_act_fun_2_classes', type=str, default="sigmoid", help='Output activation function for the two classese classification task (default: sigmoid)')
  parser.add_argument('--rpeak', action='store_true', help='R-peak pose normalization data augmentation technique')
  parser.add_argument('--different_filters', action='store_true', help='Use different filters for each lead with the 2D model')
  parser.add_argument('--num_filters_first_conv', type=int, default=1, help='Number of filters in the first convolutional layer (default: 1)')
  parser.add_argument('--one_d_model', action='store_true', help='Use the 1D model instead of the 2D model')
  parser.add_argument('--jitter_std', type=float, nargs='+', default=[0.01, 0.1], help='Interval used to extract a uniform standard deviation for the random jitter data augmentation technique (default: [0.01, 0.1])')
  
  args                                  = parser.parse_args()

  return args

#  Function useful to set the seeds (for numpy.random and tensorflow.random)
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS']  = '1'

#  Function useful to establish the output activation function of the model and to define
#  the dictionary to associate a number to each class (based on the selected number of classes).
def get_classes_dic_and_output_activation_function(num_classes):
  classes_dic_5classes                  = {"NORM": 0, "HYP": 1, "MI": 2, "STTC": 3, "CD": 4}

  classes_dic_20classes                 = {"STTC": 5, "AMI": 6, "IMI": 7, "LAFB/LPFB": 8, \
                                           "IRBBB": 9, "LVH": 10, "CLBBB": 11, "NST_": 12, "ISCA": 13, \
                                           "CRBBB": 14, "IVCD": 15, "ISC_": 16, "_AVB": 17, "ISCI": 18, \
                                           "WPW": 19, "LAO/LAE": 20, "ILBBB": 21, "RAO/RAE": 22, "LMI": 23}

  classes_dic                           = {}

  if num_classes == 5:
    classes_dic                         = {"NORM": 0, "HYP": 1, "MI": 2, "STTC": 3, "CD": 4}

  if num_classes == 20:
    classes_dic                         = {"NORM": 0, "STTC": 1, "AMI": 2, "IMI": 3, "LAFB/LPFB": 4, \
                                           "IRBBB": 5, "LVH": 6, "CLBBB": 7, "NST_": 8, "ISCA": 9, \
                                           "CRBBB": 10, "IVCD": 11, "ISC_": 12, "_AVB": 13, "ISCI": 14, \
                                           "WPW": 15, "LAO/LAE": 16, "ILBBB": 17, "RAO/RAE": 18, "LMI": 19}

  return classes_dic_5classes, classes_dic_20classes, classes_dic

#  Function useful to process and save raw data (or load it)
def process_raw_data(data_dir_exists, num_classes, classes_dic_5classes, classes_dic_20classes, classes_dic, sampling_rate, path, selected_leads_indeces, likelihood_threshold=100, train_folds=[1, 2, 3, 4, 5, 6, 7, 8], val_fold=9, test_fold=10):
  X                                     = None
  Y                                     = None
  sample_weights                        = None

  if not data_dir_exists:
    # Load and convert annotation data
    Y                                   = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes                         = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X                                   = load_raw_data(Y, sampling_rate, path, selected_leads_indeces)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df                              = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df                              = agg_df[agg_df.diagnostic == 1]

    number_of_ecgs                      = len(Y.scp_codes)

    # Apply diagnostic class/superclass
    diagnostic_superclass               = np.repeat(None, number_of_ecgs)
    sample_weights                      = np.ones(number_of_ecgs)
    folds                               = np.array(Y.strat_fold)
    for i, scp_codes, fold in zip(range(number_of_ecgs), Y.scp_codes, Y.strat_fold):
      tmp_classes                       = []

      # Filter out all those diagnostic statement with a likelihood equals to 0
      scp_codes                         = {key:val for key, val in scp_codes.items() if val != 0.0}

      less_then_100                     = any(np.fromiter(scp_codes.values(), dtype=float) < 100.0)
      equals_to_100                     = any(np.fromiter(scp_codes.values(), dtype=float) == 100.0)

      if len(scp_codes):
        if fold == val_fold or fold == test_fold:
          # Insert all the ECGs without a diagnostic statement with a likelihood equals to 100% inside the training set
          # and filter out all those diagnostic statements with a likelihood less than 100% for all those ECGs that have
          # at least one diagnostic statement with a likelihood equals to 100%.
          if less_then_100 and not equals_to_100:
            folds[i]                    = random.choice(train_folds)
          else:
            if less_then_100 and equals_to_100:
              scp_codes                 = {key:val for key, val in scp_codes.items() if val >= likelihood_threshold}

        for key, value in scp_codes.items():
          if key in agg_df.index:
            if num_classes == 2:
              tmp_classes.append(1 if agg_df.loc[key].diagnostic_class != 'NORM' else 0)
            
            if num_classes == 5:
              tmp_classes.append(classes_dic.get(agg_df.loc[key].diagnostic_class))

            if num_classes == 20:
              tmp_classes.append(classes_dic.get(agg_df.loc[key].diagnostic_subclass))

            if num_classes == 24:
              tmp_classes.append(classes_dic_5classes.get(agg_df.loc[key].diagnostic_class))
              tmp_classes.append(classes_dic_20classes.get(agg_df.loc[key].diagnostic_subclass))

        tmp_classes                     = list(set([x for x in tmp_classes if x != None]))
        
        # Filter out all those ECGs that are classified as NORM and as other not NORM diagnostic classes/superclasses.
        if len(tmp_classes) == 1 or (num_classes != 2 and len(tmp_classes) > 1 and not 0 in tmp_classes):
          diagnostic_superclass[i]      = tuple(tmp_classes)
          sample_weights[i]             = np.mean(np.fromiter(scp_codes.values(), dtype=float)) / 100
    
    Y['diagnostic_superclass']          = diagnostic_superclass
    Y['strat_fold']                     = folds

    # Clean the labels (we remove all the ECGs that are not classified into any diagnostic class/superclass)
    sample_weights                      = sample_weights[Y.diagnostic_superclass.map(lambda l: l != None)]
    X                                   = X[Y.diagnostic_superclass.map(lambda l: l != None)]
    Y                                   = Y[Y.diagnostic_superclass.map(lambda l: l != None)]

  return X, Y, sample_weights

#  Function useful to load the dataset
def load_raw_data(df, sampling_rate, path, selected_leads_indeces):
    if sampling_rate == 100:
        data                            = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data                            = [wfdb.rdsamp(path + f) for f in df.filename_hr]

    data                                = np.array([signal.transpose() for signal, meta in data])

    return data[:, selected_leads_indeces, :]

#  Function useful to split the data into train, validation and test data
def train_val_test_split(data_dir_exists, X, Y, sample_weights, num_classes, val_fold=9, test_fold=10):
  X_train_path                          = '../../data/' + str(num_classes) + 'Classes/X_train.npy'
  y_train_path                          = '../../data/' + str(num_classes) + 'Classes/y_train.pkl'
  X_val_path                            = '../../data/' + str(num_classes) + 'Classes/X_val.npy'
  y_val_path                            = '../../data/' + str(num_classes) + 'Classes/y_val.pkl'
  X_test_path                           = '../../data/' + str(num_classes) + 'Classes/X_test.npy'
  y_test_path                           = '../../data/' + str(num_classes) + 'Classes/y_test.pkl'
  sample_weights_train_path             = '../../data/' + str(num_classes) + 'Classes/sample_weights_train.npy'

  if data_dir_exists:
    X_train                             = np.load(X_train_path)
    y_train                             = pd.read_pickle(y_train_path)
    X_val                               = np.load(X_val_path)
    y_val                               = pd.read_pickle(y_val_path)
    X_test                              = np.load(X_test_path)
    y_test                              = pd.read_pickle(y_test_path)

    sample_weights_train                = np.load(sample_weights_train_path)
  else:
    #  Train
    X_train                             = X[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)]
    y_train                             = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)].diagnostic_superclass

    sample_weights_train                = sample_weights[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)]

    #  Validation
    X_val                               = X[Y.strat_fold == val_fold]
    y_val                               = Y[Y.strat_fold == val_fold].diagnostic_superclass

    #  Test
    X_test                              = X[Y.strat_fold == test_fold]
    y_test                              = Y[Y.strat_fold == test_fold].diagnostic_superclass

    np.save(X_train_path, X_train)
    y_train.to_pickle(y_train_path)
    np.save(X_val_path, X_val)
    y_val.to_pickle(y_val_path)
    np.save(X_test_path, X_test)
    y_test.to_pickle(y_test_path)
    np.save(sample_weights_train_path, sample_weights_train)

  return X_train, y_train, X_val, y_val, X_test, y_test, sample_weights_train

#  Function useful to reshape the data
def reshape_data(X_train, X_val, X_test):
  X_train                               = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
  X_val                                 = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
  X_test                                = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

  return X_train, X_val, X_test

#  Function useful to generate the average correlation matrix between the 12-leads
def correlation_matrix(X, leads, leads_name):
  X                                     = X.reshape(X.shape[0], X.shape[1], X.shape[2])
  correlation_matrix                    = np.zeros((leads, leads))

  # Compute the correlation matrix for each ECG
  for x in X:
    matrix                              = np.corrcoef(x)
    correlation_matrix                  = correlation_matrix + matrix

  # Compute the average correlation matrix
  correlation_matrix                    = correlation_matrix / X.shape[0]

  # Plot the correlation matrix
  fig, ax                               = plt.subplots(figsize = (13, 7))
  title                                 = "Correlation matrix"
  plt.title(title, fontsize = 18)
  ttl                                   = ax.title
  ttl.set_position([0.5, 1.05])

  sns.heatmap(correlation_matrix, vmin=-1, vmax=1, xticklabels=leads_name, yticklabels=leads_name, cmap='seismic', linewidths=0.30, ax=ax)
  plt.savefig("correlation_matrix.png")

#  Function useful to interpolate the data (during the data augmentation)
def interp1d(datum, new_length):
  length                                = datum.shape[1]
    
  return np.array([np.interp(np.linspace(0, length - 1, num=new_length), np.arange(length), lead[:, 0]) for lead in datum])