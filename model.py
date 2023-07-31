'''
   Deep learning project (100Hz)

   model.py

   Generate the model.

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
import tensorflow
import numpy as np
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, SpatialDropout2D, BatchNormalization, Activation, LeakyReLU, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, Adagrad, SGD
from keras.metrics import BinaryAccuracy


## Build the 2D model
def get_model_2D(init_lr, leads, num_classes, time_length, optimizer, momentum, dropout, pool_sizes, dilation_factors, kernel_size_last_conv, output_act_fun, num_filters_first_conv):
   inputs               = Input(shape=(leads, time_length, 1,), dtype="float32", name="I1")
   x                    = Conv2D(num_filters_first_conv, (1, 1), kernel_initializer="ones" if num_filters_first_conv == leads else 'glorot_uniform', name="C2")(inputs)
   x                    = BatchNormalization(name="B3")(x)
   x                    = LeakyReLU(name="A4")(x)

   x                    = Conv2D(16, (1, 5), dilation_rate=(1, dilation_factors[0]), name="C5")(x)
   x                    = BatchNormalization(name="B6")(x)
   x                    = LeakyReLU(name="A7")(x)
   x                    = MaxPooling2D((1, pool_sizes[0]), name="MP8")(x)

   x                    = Conv2D(16, (1, 5), dilation_rate=(1, dilation_factors[1]), name="C9")(x)
   x                    = BatchNormalization(name="B10")(x)
   x                    = LeakyReLU(name="A11")(x)
   x                    = MaxPooling2D((1, pool_sizes[1]), name="MP12")(x)

   x                    = Conv2D(32, (1, 5), dilation_rate=(1, dilation_factors[2]), name="C13")(x)
   x                    = BatchNormalization(name="B14")(x)
   x                    = LeakyReLU(name="A15")(x)
   x                    = MaxPooling2D((1, pool_sizes[2]), name="MP16")(x)

   x                    = Conv2D(32, (1, 3), dilation_rate=(1, dilation_factors[3]), name="C17")(x)
   x                    = BatchNormalization(name="B18")(x)
   x                    = LeakyReLU(name="A19")(x)
   x                    = MaxPooling2D((1, pool_sizes[3]), name="MP20")(x)

   x                    = Conv2D(64, (1, 3), dilation_rate=(1, dilation_factors[4]), name="C21")(x)
   x                    = BatchNormalization(name="B22")(x)
   x                    = LeakyReLU(name="A23")(x)
   x                    = MaxPooling2D((1, pool_sizes[4]), name="MP24")(x)

   x                    = Conv2D(64, (1, 3), dilation_rate=(1, dilation_factors[5]), name="C25")(x)
   x                    = BatchNormalization(name="B26")(x)
   x                    = LeakyReLU(name="A27")(x)
   x                    = MaxPooling2D((1, pool_sizes[5]), name="MP28")(x)

   x                    = SpatialDropout2D(dropout, name="Dr29")(x)

   x                    = Conv2D(128, (kernel_size_last_conv, 1), name="C30")(x)
   x                    = BatchNormalization(name="B31")(x)
   x                    = LeakyReLU(name="A32")(x)

   x                    = Flatten(name="F33")(x)

   outputs              = Dense(num_classes-1 if output_act_fun == "sigmoid" and num_classes == 2 else num_classes, activation=output_act_fun if num_classes == 2 else "sigmoid", name="D34")(x)
   model                = Model(inputs, outputs, name="CNN")

   if optimizer == "Adagrad":
      opt               = Adagrad(learning_rate=init_lr)
   elif optimizer == "SGD":
      opt               = SGD(learning_rate=init_lr, momentum=momentum)
   else:
      opt               = Adam(learning_rate=init_lr)

   # Specify the loss, optimizer, and metrics with `compile()`.
   model.compile(
      loss              = BinaryCrossentropy(),
      optimizer         = opt,
      metrics           = [keras.metrics.BinaryAccuracy()]
   )

   model.summary()

   return model



## Build the 2D model with different filters for each lead
def get_model_2D_different_filters(init_lr, leads, num_classes, time_length, optimizer, momentum, dropout, pool_sizes, dilation_factors, kernel_size_last_conv, output_act_fun, num_filters_first_conv):
   inputs               = []
   x_conv_block_x6      = []

   for lead in range(leads):
      input_lead_cluster, x_conv = get_conv_block_x6(1, lead, time_length, pool_sizes, dilation_factors)
      inputs.append(input_lead_cluster)
      x_conv_block_x6.append(x_conv)

   x                    = Concatenate(axis=1)(x_conv_block_x6)

   x                    = SpatialDropout2D(dropout, name="Dr29")(x)

   x                    = Conv2D(128, (kernel_size_last_conv, 1), name="C30")(x)
   x                    = BatchNormalization(name="B31")(x)
   x                    = LeakyReLU(name="A32")(x)

   x                    = Flatten(name="F33")(x)

   outputs              = Dense(num_classes-1 if output_act_fun == "sigmoid" and num_classes == 2 else num_classes, activation=output_act_fun if num_classes == 2 else "sigmoid", name="D34")(x)
   model                = Model(inputs, outputs, name="CNN")

   if optimizer == "Adagrad":
      opt               = Adagrad(learning_rate=init_lr)
   elif optimizer == "SGD":
      opt               = SGD(learning_rate=init_lr, momentum=momentum)
   else:
      opt               = Adam(learning_rate=init_lr)

   # Specify the loss, optimizer, and metrics with `compile()`.
   model.compile(
      loss              = BinaryCrossentropy(),
      optimizer         = opt,
      metrics           = [keras.metrics.BinaryAccuracy()]
   )

   model.summary()

   return model

# Build one of the six convolutions of the temporal fusion block
def get_conv_block_x6(leads_cluster, i, time_length, pool_sizes, dilation_factors, num_filters_first_conv):
   inputs               = Input(shape=(leads_cluster, time_length, 1,), dtype="float32", name="I1_" + str(i))
   x                    = Conv2D(num_filters_first_conv, (1, 1), kernel_initializer="ones" if num_filters_first_conv == leads else 'glorot_uniform', name="C2")(inputs)
   x                    = BatchNormalization(name="B3_" + str(i))(x)
   x                    = LeakyReLU(name="A4_" + str(i))(x)

   x                    = Conv2D(16, (1, 5), dilation_rate=(1, dilation_factors[0]), name="C5_" + str(i))(x)
   x                    = BatchNormalization(name="B6_" + str(i))(x)
   x                    = LeakyReLU(name="A7_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[0]), name="MP8_" + str(i))(x)

   x                    = Conv2D(16, (1, 5), dilation_rate=(1, dilation_factors[1]), name="C9_" + str(i))(x)
   x                    = BatchNormalization(name="B10_" + str(i))(x)
   x                    = LeakyReLU(name="A11_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[1]), name="MP12_" + str(i))(x)

   x                    = Conv2D(32, (1, 5), dilation_rate=(1, dilation_factors[2]), name="C13_" + str(i))(x)
   x                    = BatchNormalization(name="B14_" + str(i))(x)
   x                    = LeakyReLU(name="A15_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[2]), name="MP16_" + str(i))(x)

   x                    = Conv2D(32, (1, 3), dilation_rate=(1, dilation_factors[3]), name="C17_" + str(i))(x)
   x                    = BatchNormalization(name="B18_" + str(i))(x)
   x                    = LeakyReLU(name="A19_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[3]), name="MP20_" + str(i))(x)

   x                    = Conv2D(64, (1, 3), dilation_rate=(1, dilation_factors[4]), name="C21_" + str(i))(x)
   x                    = BatchNormalization(name="B22_" + str(i))(x)
   x                    = LeakyReLU(name="A23_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[4]), name="MP24_" + str(i))(x)

   x                    = Conv2D(64, (1, 3), dilation_rate=(1, dilation_factors[5]), name="C25_" + str(i))(x)
   x                    = BatchNormalization(name="B26_" + str(i))(x)
   x                    = LeakyReLU(name="A27_" + str(i))(x)
   x                    = MaxPooling2D((1, pool_sizes[5]), name="MP28_" + str(i))(x)

   return inputs, x



## Build the 1D model
def get_model_1D(init_lr, leads, num_classes, time_length, optimizer, momentum, dropout, pool_sizes, dilation_factors, kernel_size_last_conv, output_act_fun, num_filters_first_conv):
   inputs               = Input(shape=(time_length, leads,), dtype="float32", name="I1")
   x                    = Conv1D(leads, 1, kernel_initializer="ones", name="C2")(inputs)
   x                    = BatchNormalization(name="B3")(x)
   x                    = LeakyReLU(name="A4")(x)

   x                    = Conv1D(16, 5, dilation_rate=dilation_factors[0], name="C5")(x)
   x                    = BatchNormalization(name="B6")(x)
   x                    = LeakyReLU(name="A7")(x)
   x                    = MaxPooling1D(pool_sizes[0], name="MP8")(x)

   x                    = Conv1D(16, 5, dilation_rate=dilation_factors[1], name="C9")(x)
   x                    = BatchNormalization(name="B10")(x)
   x                    = LeakyReLU(name="A11")(x)
   x                    = MaxPooling1D(pool_sizes[1], name="MP12")(x)

   x                    = Conv1D(32, 5, dilation_rate=dilation_factors[2], name="C13")(x)
   x                    = BatchNormalization(name="B14")(x)
   x                    = LeakyReLU(name="A15")(x)
   x                    = MaxPooling1D(pool_sizes[2], name="MP16")(x)

   x                    = Conv1D(32, 3, dilation_rate=dilation_factors[3], name="C17")(x)
   x                    = BatchNormalization(name="B18")(x)
   x                    = LeakyReLU(name="A19")(x)
   x                    = MaxPooling1D(pool_sizes[3], name="MP20")(x)

   x                    = Conv1D(64, 3, dilation_rate=dilation_factors[4], name="C21")(x)
   x                    = BatchNormalization(name="B22")(x)
   x                    = LeakyReLU(name="A23")(x)
   x                    = MaxPooling1D(pool_sizes[4], name="MP24")(x)

   x                    = Conv1D(64, 3, dilation_rate=dilation_factors[5], name="C25")(x)
   x                    = BatchNormalization(name="B26")(x)
   x                    = LeakyReLU(name="A27")(x)
   x                    = MaxPooling1D(pool_sizes[5], name="MP28")(x)

   x                    = SpatialDropout1D(dropout, name="Dr29")(x)

   x                    = Conv1D(128, kernel_size_last_conv, name="C30")(x)
   x                    = BatchNormalization(name="B31")(x)
   x                    = LeakyReLU(name="A32")(x)

   x                    = Flatten(name="F33")(x)

   outputs              = Dense(num_classes-1 if output_act_fun == "sigmoid" and num_classes == 2 else num_classes, activation=output_act_fun if num_classes == 2 else "sigmoid", name="D34")(x)
   model                = Model(inputs, outputs, name="CNN")

   if optimizer == "Adagrad":
      opt               = Adagrad(learning_rate=init_lr)
   elif optimizer == "SGD":
      opt               = SGD(learning_rate=init_lr, momentum=momentum)
   else:
      opt               = Adam(learning_rate=init_lr)

   # Specify the loss, optimizer, and metrics with `compile()`.
   model.compile(
      loss              = BinaryCrossentropy(),
      optimizer         = opt,
      metrics           = [keras.metrics.BinaryAccuracy()]
   )

   model.summary()

   return model