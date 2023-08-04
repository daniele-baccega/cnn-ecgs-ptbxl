'''
  Deep learning project (100Hz)

  datagenerator.py

  Data generator used for data augmentation techniques (random jitter, amplitude transformation, time scale transformation, random cropping) and
  to standardize and pad the ECGs.

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
import numpy as np
import neurokit2 as nk
import os


## Import utils
from utils import interp1d


## Data generator
def dataGenerator(sampling_rate, num_classes, output_act_fun, means, stds, sample_weights, data, labels, batchSize=32, one_d_model=False, rpeak=False, different_filters=False, save_test=False, cropLength=344, padding=0, jitter=[0, 0], multiply_amplitude=[1, 1], new_scale=[1, 1], nSamples=0):
  test_data_dir                   = 'test_data'
  i                               = 0

  while True:
    # We take all the samples available by default, without guarantees each batch containing exactly batchSize samples
    if (nSamples == 0):
      nSamples                    = data.shape[0]
    
    for start in range(0, nSamples, batchSize):
      x_batch                     = []
      y_batch                     = []
      
      end                         = min(start + batchSize, nSamples)

      dataBatch                   = data[start:end]
      labelsBatch                 = labels[start:end]

      for datum in dataBatch:
        # Normal random jitter
        if jitter:
          jitter_std              = np.random.uniform(jitter[0], jitter[1])
          jitter_value            = np.random.normal(0.0, jitter_std, (datum.shape[0], datum.shape[1], 1))
          datum                   = datum + jitter_value

        # Standardization
        datum                     = (datum - means) / stds

        # Amplitude transformation
        alfa                      = np.random.uniform(multiply_amplitude[0], multiply_amplitude[1])
        datum                     = datum * alfa

        # Time scale transformation
        scale                     = np.random.uniform(new_scale[0], new_scale[1])
        datum                     = interp1d(datum, int(datum.shape[1] * scale))
        datum                     = datum.reshape(datum.shape[0], datum.shape[1], 1)
        
        if cropLength > 0:
          if datum.shape[1] < cropLength:
            print("ERROR time series " + datum + " length is " + str(datum.shape[1]) + " < " + str(cropLength))
            os.sys.exit(1)

          if rpeak:
            _, rpeaks             = nk.ecg_peaks(datum[0, :, 0], sampling_rate=sampling_rate*scale)

            rpeaks['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'][np.where((rpeaks['ECG_R_Peaks'] - (cropLength / 2)) >= 0)]
            rpeaks['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'][np.where((rpeaks['ECG_R_Peaks'] + (cropLength / 2)) <= datum.shape[1])]

            if rpeaks['ECG_R_Peaks'] != []:
              random_rpeak        = np.random.choice(rpeaks['ECG_R_Peaks'])
            else:
              random_rpeak        = np.random.randint(cropLength // 2, datum.shape[1] - cropLength // 2)

            init                  = random_rpeak - (cropLength // 2)
            stop                  = random_rpeak + (cropLength // 2)
          else:
            init                  = np.random.randint(0, datum.shape[1] - cropLength)
            stop                  = init + cropLength

          # Cropping
          datum                   = datum[:, init:stop, 0]

        # Padding
        datum                     = [np.pad(lead, (padding, padding), 'constant', constant_values=(0, 0)) for lead in datum]
        datum                     = np.array(datum)
        datum                     = datum.reshape(datum.shape[0], datum.shape[1])

        # Posiional encoding
        # ...

        if one_d_model:
          x_batch.append(datum.transpose())
        else:
          x_batch.append(datum)
      
      y_batch.append(labelsBatch)

      x_batch                     = np.array(x_batch)
      y_batch                     = np.array(y_batch)
      y_batch                     = y_batch.reshape(y_batch.shape[1], num_classes-1 if output_act_fun == 'sigmoid' and num_classes == 2 else num_classes)

      if save_test:
        if not os.path.exists(test_data_dir):
          os.mkdir(test_data_dir)

        np.save(test_data_dir + '/test_' + str(i) + '.npy', x_batch)
        i                         = i + 1

      if different_filters:
        yield [x_batch[:, 0, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 1, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 2, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 3, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 4, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]), 
               x_batch[:, 5, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 6, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]),
               x_batch[:, 7, :].reshape(x_batch.shape[0], 1, x_batch.shape[2])], y_batch, sample_weights[start:end]
      else:
        yield x_batch, y_batch, sample_weights[start:end]