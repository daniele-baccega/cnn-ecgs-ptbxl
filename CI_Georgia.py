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
import physionet_challenge_utility_script as pc
from sklearn.metrics import auc, roc_curve, multilabel_confusion_matrix


paths 							= ["GeorgiaRefinement/D1", "GeorgiaRefinement/D1-D2", "GeorgiaRefinement/D1-V1", "GeorgiaRefinement/D1-V2","GeorgiaRefinement/D1-V3", "GeorgiaRefinement/D1-V4", "GeorgiaRefinement/D1-V5", "GeorgiaRefinement/D1-V6", "GeorgiaRefinement/8leads", "GeorgiaRefinement/12leads", "GeorgiaRefinement/12leads_WithoutDataAugmentation"]

path							= 'Georgia/'
num_classes 						= 20
first_iteration 					= True

_, _, labels, ecg_filenames               = pc.import_key_data(path)

SNOMED_scored                             = pd.read_csv("SNOMED_mappings_scored.csv", sep=",")
SNOMED_unscored                           = pd.read_csv("SNOMED_mappings_unscored.csv", sep=",")
df_labels                                 = pc.make_undefined_class(labels, SNOMED_unscored)

SNOMED_dic                                = dict()
for _, row in SNOMED_scored.iterrows():
  SNOMED_dic[str(row["SNOMED CT Code"])]  = row["Abbreviation"]

classes_dic_name_id                               = dict()
i                                         = 0
for value in SNOMED_dic.values():
  if value not in classes_dic_name_id.keys():
    classes_dic_name_id[value]                    = i
    i                                     = i + 1

classes_dic_name_id["Average"] = i

classes_dic = dict()
for key, value in classes_dic_name_id.items():
	classes_dic[value] = key

num_classes                               = len(classes_dic.keys())-1

global_roc_auc_mean 					= np.zeros((num_classes+1, len(paths)))
global_roc_auc_left 					= np.zeros((num_classes+1, len(paths)))
global_roc_auc_right 					= np.zeros((num_classes+1, len(paths)))
j							= 0
for path in paths:
	roc_auc_mean 					= np.zeros(num_classes+1)
	roc_auc_variance 				= np.zeros(num_classes+1)

	runs 						= len(fnmatch.filter(os.listdir(path + "/"), '20Classes_*')) + 1

	for count in range(1, runs):
		roc_auc_local 				= np.zeros(num_classes+1)

		#  Load the files
		with open(path + '/20Classes_' + str(count-1) + '/y_pred_Georgia', 'rb') as y_pred_file:
			y_pred 				= pickle.load(y_pred_file)

		with open(path + '/20Classes_' + str(count-1) + '/y_test_Georgia', 'rb') as y_test_file:
			y_test 				= pickle.load(y_test_file)

		#  Plot ROC curves
		fpr 					= dict()
		tpr 					= dict()

		for i in range(num_classes):
		  	fpr[i], tpr[i], _ 		= roc_curve(y_test[:, i], y_pred[:, i])
		  	roc_auc_local[i] 		= auc(fpr[i], tpr[i])

		roc_auc_local[21] 			= np.mean(roc_auc_local[0:21])

		y_pred 					= np.where(y_pred > 0.5, 1, 0)

		if first_iteration:
			roc_auc_mean 			= roc_auc_local
			first_iteration 		= False
		else:
			roc_auc_mean 			= roc_auc_mean + (roc_auc_local - roc_auc_mean) / count
			roc_auc_variance 		= roc_auc_variance + ((count - 1) / count) * (roc_auc_local - roc_auc_mean) ** 2


	roc_auc_std 					= np.sqrt(roc_auc_variance / (count - 1))

	roc_auc_left 					= roc_auc_mean - 1.96 * (roc_auc_std / math.sqrt(count))
	roc_auc_right 					= roc_auc_mean + 1.96 * (roc_auc_std / math.sqrt(count))

	print("\n" + path + ":")
	print("AUC:")
	for i in range(num_classes+1):
		print("{0}: {1:0.5f} {2:0.5f} {3:0.5f} (±{4:0.5f})".format(classes_dic.get(i), roc_auc_left[i]*100, roc_auc_mean[i]*100, roc_auc_right[i]*100, (1.96 * (roc_auc_std[i] / math.sqrt(count)))*100))

	global_roc_auc_mean[:, j]			= roc_auc_mean * 100
	global_roc_auc_left[:, j]			= roc_auc_left * 100
	global_roc_auc_right[:, j]			= roc_auc_right * 100

	j 						= j + 1


pd.DataFrame(global_roc_auc_mean, index=classes_dic_name_id.values(), columns=paths).to_csv("mean_AUC_Georgia.csv", float_format='%.2f')
pd.DataFrame(global_roc_auc_left, index=classes_dic_name_id.values(), columns=paths).to_csv("left_AUC_Georgia.csv", float_format='%.2f')
pd.DataFrame(global_roc_auc_right, index=classes_dic_name_id.values(), columns=paths).to_csv("right_AUC_Georgia.csv", float_format='%.2f')
