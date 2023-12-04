'''
	Deep learning project (100Hz)

	CI.py

	Compute the averages and the 95% CIs for the AUC, sensitivity and specificity.

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
import pandas as pd
import argparse
import pickle
import math
import os
import fnmatch
from sklearn.metrics import auc, roc_curve, multilabel_confusion_matrix


paths 								= ["D1", "D1-D2", "D1-V1", "D1-V2","D1-V3", "D1-V4", "D1-V5", "D1-V6", "8leads", "12leads", "12leads_WithoutDataAugmentation"]

num_classes 						= 20
first_iteration 					= True

classes_dic			 				= {0: "NORM", 1: "STTC", 2: "AMI", 3: "IMI", 4: "LAFB/LPFB", \
				               		   5: "IRBBB", 6: "LVH", 7: "CLBBB", 8: "NST_", 9: "ISCA", \
				               		   10: "CRBBB", 11: "IVCD", 12: "ISC_", 13: "_AVB", 14: "ISCI", \
				               		   15: "WPW", 16: "LAO/LAE", 17: "ILBBB", 18: "RAO/RAE", 19: "LMI", \
				               		   20: "Average"}

global_roc_auc_mean 				= np.zeros((num_classes+1, len(paths)))
global_sensitivity_mean 			= np.zeros((num_classes+1, len(paths)))
global_specificity_mean 			= np.zeros((num_classes+1, len(paths)))
global_roc_auc_left 				= np.zeros((num_classes+1, len(paths)))
global_sensitivity_left 			= np.zeros((num_classes+1, len(paths)))
global_specificity_left 			= np.zeros((num_classes+1, len(paths)))
global_roc_auc_right 				= np.zeros((num_classes+1, len(paths)))
global_sensitivity_right 			= np.zeros((num_classes+1, len(paths)))
global_specificity_right 			= np.zeros((num_classes+1, len(paths)))
j									= 0
for path in paths:
	roc_auc_mean 					= np.zeros(num_classes+1)
	sensitivity_mean				= np.zeros(num_classes+1)
	specificity_mean				= np.zeros(num_classes+1)
	roc_auc_variance 				= np.zeros(num_classes+1)
	sensitivity_variance			= np.zeros(num_classes+1)
	specificity_variance			= np.zeros(num_classes+1)

	runs 							= len(fnmatch.filter(os.listdir(path + "/"), '20Classes_*')) + 1

	for count in range(1, runs):
		roc_auc_local 				= np.zeros(num_classes+1)
		sensitivity_local 			= np.zeros(num_classes+1)
		specificity_local 			= np.zeros(num_classes+1)

		#  Load the files
		with open(path + '/20Classes_' + str(count-1) + '/y_pred_Georgia', 'rb') as y_pred_file:
			y_pred 					= pickle.load(y_pred_file)

		with open(path + '/20Classes_' + str(count-1) + '/y_test_Georgia', 'rb') as y_test_file:
			y_test 					= pickle.load(y_test_file)

		#  Plot ROC curves
		fpr 						= dict()
		tpr 						= dict()

		for i in range(num_classes):
		  	fpr[i], tpr[i], _ 		= roc_curve(y_test[:, i], y_pred[:, i])
		  	roc_auc_local[i] 		= auc(fpr[i], tpr[i])

		roc_auc_local[20] 			= np.mean(roc_auc_local[0:20])

		y_pred 						= np.where(y_pred > 0.5, 1, 0)

		mcm 						= multilabel_confusion_matrix(y_test, y_pred)
		tn 							= mcm[:, 0, 0]
		tp 							= mcm[:, 1, 1]
		fn 							= mcm[:, 1, 0]
		fp 							= mcm[:, 0, 1]
		sensitivity_local[0:20]		= tp / (tp + fn)
		specificity_local[0:20]		= tn / (tn + fp)
		sensitivity_local[20]		= np.mean(sensitivity_local[0:20])
		specificity_local[20]		= np.mean(specificity_local[0:20])

		if first_iteration:
			roc_auc_mean 			= roc_auc_local
			sensitivity_mean		= sensitivity_local
			specificity_mean    	= specificity_local
			first_iteration 		= False
		else:
			roc_auc_mean 			= roc_auc_mean + (roc_auc_local - roc_auc_mean) / count
			sensitivity_mean		= sensitivity_mean + (sensitivity_local - sensitivity_mean) / count
			specificity_mean		= specificity_mean + (specificity_local - specificity_mean) / count
			roc_auc_variance 		= roc_auc_variance + ((count - 1) / count) * (roc_auc_local - roc_auc_mean) ** 2
			sensitivity_variance 	= sensitivity_variance + ((count - 1) / count) * (sensitivity_local - sensitivity_mean) ** 2
			specificity_variance 	= specificity_variance + ((count - 1) / count) * (specificity_local - specificity_mean) ** 2


	roc_auc_std 					= np.sqrt(roc_auc_variance / (count - 1))
	sensitivity_std 				= np.sqrt(sensitivity_variance / (count - 1))
	specificity_std 				= np.sqrt(specificity_variance / (count - 1))

	roc_auc_left 					= roc_auc_mean - 1.96 * (roc_auc_std / math.sqrt(count))
	sensitivity_left				= sensitivity_mean - 1.96 * (sensitivity_std / math.sqrt(count))
	specificity_left				= specificity_mean - 1.96 * (specificity_std / math.sqrt(count))
	roc_auc_right 					= roc_auc_mean + 1.96 * (roc_auc_std / math.sqrt(count))
	sensitivity_right				= sensitivity_mean + 1.96 * (sensitivity_std / math.sqrt(count))
	specificity_right				= specificity_mean + 1.96 * (specificity_std / math.sqrt(count))

	print("\n" + path + ":")
	print("AUC:")
	for i in range(num_classes+1):
		print("{0}: {1:0.5f} {2:0.5f} {3:0.5f} (±{4:0.5f})".format(classes_dic.get(i), roc_auc_left[i]*100, roc_auc_mean[i]*100, roc_auc_right[i]*100, (1.96 * (roc_auc_std[i] / math.sqrt(count)))*100))

	print("\nSensitivity:")

	for i in range(num_classes+1):
		print("{0}: {1:0.5f} {2:0.5f} {3:0.5f} (±{4:0.5f})".format(classes_dic.get(i), sensitivity_left[i]*100, sensitivity_mean[i]*100, sensitivity_right[i]*100, (1.96 * (sensitivity_std[i] / math.sqrt(count)))*100))

	print("\nSpecificity:")

	for i in range(num_classes+1):
		print("{0}: {1:0.5f} {2:0.5f} {3:0.5f} (±{4:0.5f})".format(classes_dic.get(i), specificity_left[i]*100, specificity_mean[i]*100, specificity_right[i]*100, (1.96 * (specificity_std[i] / math.sqrt(count)))*100))

	global_roc_auc_mean[:, j]		= roc_auc_mean * 100
	global_sensitivity_mean[:, j]	= sensitivity_mean * 100
	global_specificity_mean[:, j]	= specificity_mean * 100
	global_roc_auc_left[:, j]		= roc_auc_left * 100
	global_sensitivity_left[:, j]	= sensitivity_left * 100
	global_specificity_left[:, j]	= specificity_left * 100
	global_roc_auc_right[:, j]		= roc_auc_right * 100
	global_sensitivity_right[:, j]	= sensitivity_right * 100
	global_specificity_right[:, j]	= specificity_right * 100

	j 								= j + 1


pd.DataFrame(global_roc_auc_mean, index=classes_dic.values(), columns=paths).to_csv("mean_AUC.csv", float_format='%.2f')
pd.DataFrame(global_roc_auc_left, index=classes_dic.values(), columns=paths).to_csv("left_AUC.csv", float_format='%.2f')
pd.DataFrame(global_roc_auc_right, index=classes_dic.values(), columns=paths).to_csv("right_AUC.csv", float_format='%.2f')

pd.DataFrame(global_sensitivity_mean, index=classes_dic.values(), columns=paths).to_csv("mean_sensitivity.csv", float_format='%.2f')
pd.DataFrame(global_sensitivity_left, index=classes_dic.values(), columns=paths).to_csv("left_sensitivity.csv", float_format='%.2f')
pd.DataFrame(global_sensitivity_right, index=classes_dic.values(), columns=paths).to_csv("right_sensitivity.csv", float_format='%.2f')

pd.DataFrame(global_specificity_mean, index=classes_dic.values(), columns=paths).to_csv("mean_specificity.csv", float_format='%.2f')
pd.DataFrame(global_specificity_left, index=classes_dic.values(), columns=paths).to_csv("left_specificity.csv", float_format='%.2f')
pd.DataFrame(global_specificity_right, index=classes_dic.values(), columns=paths).to_csv("right_specificity.csv", float_format='%.2f')