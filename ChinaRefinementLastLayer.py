'''
  Deep learning project (100Hz)

  ChinaRefinementLastLayer.py

  Fine-tuned the classification layer of the orignal network using the China dataset train set.

  Inputs:
     --seed:                     random seed
     --scenario:                 scenario simulated
     --path:                     path to the directory in which to load the pre-trained model
     --newpath:                  path to the directory in which to save the fine-tuned models

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
import argparse
import numpy as np
import pandas as pd
import pickle
import physionet_challenge_utility_script as pc
import tensorflow
from plotnine import *
from tensorflow import keras
from keras import models
from keras import Model
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
from skmultilearn.model_selection import iterative_train_test_split


## Import data generator and utils
from datagenerator import dataGenerator
from utils import interp1d

## Verify what tensorflow version is used
print("Using tensorflow version " + str(tensorflow.__version__))
print("Using keras version " + str(tensorflow.keras.__version__))

#  This prevents tensorflow from allocating all memory on GPU - for TensorFlow 2.2+
gpus                                                        = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tensorflow.config.experimental.set_memory_growth(gpu, True)


## Parse the arguments
parser                                                      = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default="123456789", help='Seed (default: 123456789)')
parser.add_argument('--scenario', type=str, default="D1", help='Scenario simulated, must be a directory name (default: D1)')
parser.add_argument('--path', type=str, default="GeorgiaRefinementAll/D1/20Classes_0", help='Path to the run directory (default: GeorgiaRefinementAll/D1/20Classes_0)')
parser.add_argument('--newpath', type=str, default="ChinaRefinementLastLayer/D1/20Classes_0", help='Path to the directory in which to save the refined models (default: ChinaRefinementLastLayer/D1/20Classes_0)')

args                                                        = parser.parse_args()

 
## Initialize some variables
path                                                        = 'China/'
activation_function                                         = 'sigmoid'
sampling_rate                                               = 100
test_proportion                                             = 0.2
batch_size                                                  = 32
crop_window                                                 = 344
jitter_std                                                  = [0.01, 0.1]
amplitude_scale                                             = [0.7, 1.3]
time_scale                                                  = [0.8, 1.2]
epochs                                                      = 10

leads_dict                                                  = {"D1": ["I"],
                                                               "D1-D2": ["I", "II"],
                                                               "D1-V1": ["I", "V1"],
                                                               "D1-V2": ["I", "V2"],
                                                               "D1-V3": ["I", "V3"],
                                                               "D1-V4": ["I", "V4"],
                                                               "D1-V5": ["I", "V5"],
                                                               "D1-V6": ["I", "V6"],
                                                               "8leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
                                                               "12leads": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                                                               "12leads_WithoutDataAugmentation": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]}


## Load the scored diagnostic classes
_, _, labels, ecg_filenames                                 = pc.import_key_data_China(path)

SNOMED_scored                                               = pd.read_csv("SNOMED_mappings_scored_China.csv", sep=",")
SNOMED_unscored                                             = pd.read_csv("SNOMED_mappings_unscored_China.csv", sep=",")
df_labels                                                   = pc.make_undefined_class(labels, SNOMED_unscored)

y, snomed_classes                                           = pc.onehot_encode(df_labels)

SNOMED_dic                                                  = dict()
for _, row in SNOMED_scored.iterrows():
  SNOMED_dic[str(row["SNOMED CT Code"])]                    = row["Abbreviation"]

classes_dic                                                 = dict()
i                                                           = 0
for value in SNOMED_dic.values():
  if value not in classes_dic.keys():
    classes_dic[value]                                      = i
    i                                                       = i + 1

num_classes                                                 = len(classes_dic)

y_data                                                      = np.zeros((len(ecg_filenames), num_classes), dtype=int)
for i in range(len(ecg_filenames)):
  for j in range(len(snomed_classes)):
    if y[i][j] == 1:
      y_data[i][classes_dic[SNOMED_dic[snomed_classes[j]]]] = 1

data                                                        = []
for ecg in ecg_filenames:
  signals, fields                                           = pc.load_challenge_data(ecg)
  init                                                      = np.random.randint(0, signals.shape[1] - 5000)
  stop                                                      = init + 5000
  signals                                                   = signals[:, init:stop]
  signals                                                   = signals.reshape(signals.shape[0], signals.shape[1], 1)
  signals                                                   = interp1d(signals, int(signals.shape[1] / 5))
  signals                                                   = signals / 100
  data.append(signals)

data                                                        = np.array(data)
y_data                                                      = np.array(y_data)

print("Data shape: ", data.shape)
print("Labels shape: ", y_data.shape)

leads                                                       = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Select the leads
selected_leads_indeces                                      = [i for i in range(0, len(leads)) if leads[i] in leads_dict[args.scenario]]
selected_leads_name                                         = [leads[i] for i in selected_leads_indeces]

data                                                        = data[:, selected_leads_indeces, :]
data                                                        = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

## Train/Test split
X_train, y_train, X_test, y_test                            = iterative_train_test_split(data, y_data, test_size=test_proportion)


# Load means and stds
with open('TrainedModels/' + args.scenario + '/means', 'rb') as means_file:
  means                                                     = pickle.load(means_file)[selected_leads_indeces, :, 0]

with open('TrainedModels/' + args.scenario + '/stds', 'rb') as stds_file:
  stds                                                      = pickle.load(stds_file)[selected_leads_indeces, :, 0]

#  Load the model at the last epoch
model                                                       = models.load_model(args.path + '/checkpoints/model_last_epoch.h5')
model.trainable                                             = False

new_classifier                                              = Dense(num_classes, activation=activation_function, name="D34")(model.layers[-2].output)

model                                                       = Model(inputs=model.inputs, outputs=new_classifier, name="CNN")

# Specify the loss, optimizer, and metrics with `compile()`.
model.compile(
  loss                                                      = BinaryCrossentropy(),
  optimizer                                                 = Adam(learning_rate=1e-3),
  metrics                                                   = [BinaryAccuracy()]
)

model.summary()

sample_weights_train                                        = np.ones(X_train.shape[0])
sample_weights_test                                         = np.ones(X_test.shape[0])

#  Train the model
history                                                     = model.fit(dataGenerator(sampling_rate,
                                                                                      num_classes,
                                                                                      activation_function,
                                                                                      means,
                                                                                      stds,
                                                                                      sample_weights_train,
                                                                                      X_train,
                                                                                      y_train,
                                                                                      batch_size,
                                                                                      False,
                                                                                      False,
                                                                                      False,
                                                                                      False,
                                                                                      crop_window,
                                                                                      0,
                                                                                      jitter_std,
                                                                                      amplitude_scale,
                                                                                      time_scale),
                                                                        steps_per_epoch   = X_train.shape[0] // batch_size,
                                                                        epochs            = epochs,
					                                                              validation_data   = dataGenerator(sampling_rate,
                                                                                                          num_classes,
                                                                                                          activation_function,
                                                                                                          means,
                                                                                                          stds,
                                                                                                          sample_weights_test,
                                                                                                          X_test,
                                                                                                          y_test,
                                                                                                          batch_size,
                                                                                                          False,
                                                                                                          False,
                                                                                                          False,
                                                                                                          False,
                                                                                                          crop_window,
                                                                                                          0),
                                                                        validation_steps  = X_test.shape[0] // batch_size,
                                                                        shuffle           = True,
                                                                        workers           = 1,
                                                                        verbose           = 1)

#  Save the model at the last epoch
model.save(args.newpath + "/checkpoints/model_last_epoch.h5")

#  Plot losses and accuracies
history_df 													                        = pd.DataFrame(columns=['epoch', 'loss', 'type'])

history_df['epoch'] 								                        = np.concatenate((np.linspace(1, epochs, epochs, dtype=int), np.linspace(1, epochs, epochs, dtype=int)))
history_df['loss'] 									                        = np.concatenate((history.history['loss'], history.history['val_loss']))
history_df['type'] 									                        = np.concatenate((["Train" for e in range(epochs)], ["Val" for e in range(epochs)]))

my_plot_history_loss 								                        = (ggplot(history_df) \
																				                        + aes(x='epoch', y = 'loss', color = 'type') \
																				                        + geom_line() \
																				                        + labs(title = "Loss", x = 'epoch', y = 'loss', color = 'Type')) \
																				                        + scale_color_manual(values=['#FF0000', '#0000FF']) \
																				                        + theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold"))
	
my_plot_history_loss.save(args.newpath + '/loss', dpi=600)

history_df 													                        = pd.DataFrame(columns=['epoch', 'accuracy', 'type'])

history_df['epoch'] 								                        = np.concatenate((np.linspace(1, epochs, epochs, dtype=int), np.linspace(1, epochs, epochs, dtype=int)))
history_df['accuracy'] 							                        = np.concatenate((history.history['binary_accuracy'], history.history['val_binary_accuracy']))
history_df['type'] 									                        = np.concatenate((["Train" for e in range(epochs)], ["Val" for e in range(epochs)]))

my_plot_history_accuracy 						                        = (ggplot(history_df) \
																				                        + aes(x='epoch', y = 'accuracy', color = 'type') \
																				                        + geom_line() \
																				                        + labs(title = "Accuracy", x = 'epoch', y = 'accuracy', color = 'Type')) \
																				                        + scale_color_manual(values=['#FF0000', '#0000FF']) \
																				                        + theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold"))
	
my_plot_history_accuracy.save(args.newpath + '/accuracy', dpi=600)

#  Predict the labels of the data inside the test set and save the predictions
y_pred                                                      = model.predict(dataGenerator(sampling_rate,
							                                                                            num_classes,
                                                                                          activation_function,
                                                                                          means,
                                                                                          stds,
                                                                                          sample_weights_test,
                                                                                          X_test,
                                                                                          y_test,
                                                                                          1),
                                                                            steps   = X_test.shape[0],
                                                                            workers = 1,
                                                                            verbose = 1)

#  Save the predictions
with open(args.newpath + '/y_pred_China', 'wb') as y_pred_file:
  pickle.dump(y_pred, y_pred_file)

with open(args.newpath + '/y_test_China', 'wb') as y_test_file:
  pickle.dump(y_test, y_test_file)
