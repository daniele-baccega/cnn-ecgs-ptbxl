'''
	Deep learning project (100Hz)

	process_results.py

	Inputs:
	  --num_classes:              classification task (binary classification, five-class multi-label classification or
	                              twenty-class multi-label classification)
	  --out_act_fun_2_classes:    output activation function for the two classese classification task

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
import pandas as pd
import numpy as np
import pickle
import math
import os
from sklearn.metrics import auc, roc_curve, accuracy_score


num_classes = 20
first_iteration = True

classes_dic = {0: "NORM", 1: "STTC", 2: "AMI", 3: "IMI", 4: "LAFB/LPFB", \
               5: "IRBBB", 6: "LVH", 7: "CLBBB", 8: "NST_", 9: "ISCA", \
               10: "CRBBB", 11: "IVCD", 12: "ISC_", 13: "_AVB", 14: "ISCI", \
               15: "WPW", 16: "LAO/LAE", 17: "ILBBB", 18: "RAO/RAE", 19: "LMI"}

accuracy_mean = np.zeros(num_classes)
accuracy_variance = np.zeros(num_classes)
roc_auc_mean = np.zeros(num_classes)
roc_auc_variance = np.zeros(num_classes)

for count in range(1, 51):
	accuracy_local = np.zeros(num_classes)
	roc_auc_local = np.zeros(num_classes)

	#  Load the files
	with open('20Classes_' + str(count-1) + '/y_pred', 'rb') as y_pred_file:
		y_pred = pickle.load(y_pred_file)

	with open('20Classes_' + str(count-1) + '/y_test', 'rb') as y_test_file:
		y_test = pickle.load(y_test_file)

	#  Plot ROC curves
	fpr = dict()
	tpr = dict()

	for i in range(num_classes):
	  	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
	  	roc_auc_local[i] = auc(fpr[i], tpr[i])

	y_pred = np.where(y_pred > 0.5, 1, 0)

	for i in range(num_classes):
		accuracy_local[i] = accuracy_score(y_test[:, i], y_pred[:, i])

	if first_iteration:
		accuracy_mean = accuracy_local
		roc_auc_mean = roc_auc_local

		first_iteration = False
	else:
		accuracy_mean = accuracy_mean + (accuracy_local - accuracy_mean) / count
		roc_auc_mean = roc_auc_mean + (roc_auc_local - roc_auc_mean) / count

		accuracy_variance = accuracy_variance + ((count - 1) / count) * (accuracy_local - accuracy_mean) ** 2
		roc_auc_variance = roc_auc_variance + ((count - 1) / count) * (roc_auc_local - roc_auc_mean) ** 2


accuracy_std = np.sqrt(accuracy_variance / (count - 1))
roc_auc_std = np.sqrt(roc_auc_variance / (count - 1))

accuracy_left = accuracy_mean - 1.96 * (accuracy_std / math.sqrt(count))
accuracy_right = accuracy_mean + 1.96 * (accuracy_std / math.sqrt(count))

roc_auc_left = roc_auc_mean - 1.96 * (roc_auc_std / math.sqrt(count))
roc_auc_right = roc_auc_mean + 1.96 * (roc_auc_std / math.sqrt(count))

for i in range(num_classes):
	#print("Accuracy confidence interval class {0}: {1:0.5f} {2:0.5f} {3:0.5f}".format(classes_dic.get(i), accuracy_left[i], accuracy_mean[i], accuracy_right[i]))
	print("AUC confidence interval class {0}: {1:0.5f} {2:0.5f} {3:0.5f}".format(classes_dic.get(i), roc_auc_left[i], roc_auc_mean[i], roc_auc_right[i]))

#print("\n\nAccuracy interval:\n {0}".format(1.96 * (accuracy_std / math.sqrt(count))))
print("AUC interval:\n {0}".format(1.96 * (roc_auc_std / math.sqrt(count))))

print("AUC mean: {0:0.5f}".format(np.mean(roc_auc_mean)))