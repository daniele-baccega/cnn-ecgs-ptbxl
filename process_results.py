'''
	Deep learning project (100Hz)

	process_results.py

	Inputs:
	  --num_classes:              classification task (binary classification, five-class multi-label classification or
	                              twenty-class multi-label classification)
	  --out_act_fun_2_classes:    output activation function for the two classese classification task
	  --sigmoid_threshold:			  threshold used with the sigmoid activation function

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
import argparse
from plotnine import *
from sklearn.metrics import auc, classification_report, multilabel_confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score


## Import cf_matrix
from cf_matrix import make_confusion_matrix


## Parse the arguments
parser 															= argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (default: 2)')
parser.add_argument('--out_act_fun_2_classes', type=str, default="sigmoid", help='Output activation function for the two classese classification task (default: sigmoid)')
parser.add_argument('--sigmoid_threshold', type=float, default=0.5, help='Threshold used with the sigmoid activation function (default: 0.5)')

args 																= parser.parse_args()

if args.num_classes not in [2, 5, 20, 24]:
  print("The number of classes must be equals to 2, 5, 20 or 24.")
  exit()

if args.out_act_fun_2_classes not in ["sigmoid", "softmax"]:
  print("The output activation function for the two classes classification task must be equals to sigmoid or softmax.")
  exit()


## Compute the metrics and generate the plots for the selected task
dir_name 														= str(args.num_classes) + "Classes" 
print(str(args.num_classes) + " classes:")


#  Load the files
with open(dir_name + '/y_pred', 'rb') as y_pred_file:
	y_pred 														= pickle.load(y_pred_file)

with open(dir_name + '/y_test', 'rb') as y_test_file:
	y_test 														= pickle.load(y_test_file)

with open(dir_name + '/history', 'rb') as history_file:
	history 													= pickle.load(history_file)

ylims_acc 													= [0.7, 0.92]
ylims_loss 													= [0.1, 0.6]

classes_dic													= {0: "NORM", 1: "Not NORM"}

if args.num_classes == 5:
	classes_dic 											= {0: "NORM", 1: "HYP", 2: "MI", 3: "STTC", 4: "CD"}
	ylims_acc 												= [0.78, 0.92]

#  Subclasses that we didn't consider: RVH, SEHYP, PMI.
if args.num_classes == 20:
	classes_dic 											= {0: "NORM", 1: "STTC", 2: "AMI", 3: "IMI", 4: "LAFB/LPFB", \
										                   5: "IRBBB", 6: "LVH", 7: "CLBBB", 8: "NST_", 9: "ISCA", \
										                   10: "CRBBB", 11: "IVCD", 12: "ISC_", 13: "_AVB", 14: "ISCI", \
										                   15: "WPW", 16: "LAO/LAE", 17: "ILBBB", 18: "RAO/RAE", 19: "LMI"}
	ylims_acc 												= [0.9, 0.97]
	ylims_loss 												= [0.05, 0.3]

if args.num_classes == 24:
	classes_dic 											= {0: "NORM", 1: "HYP", 2: "MI", 3: "STTC", 4: "CD", \
																			 5: "STTC", 6: "AMI", 7: "IMI", 8: "LAFB/LPFB", \
											                 9: "IRBBB", 10: "LVH", 11: "CLBBB", 12: "NST_", 13: "ISCA", \
											                 14: "CRBBB", 15: "IVCD", 16: "ISC_", 17: "_AVB", 18: "ISCI", \
											                 19: "WPW", 20: "LAO/LAE", 21: "ILBBB", 22: "RAO/RAE", 23: "LMI"}
	ylims_acc										 			= [0.6, 0.98]
	ylims_loss 												= [0.05, 1]


min_train_loss_idx 									= np.where(history['loss']==np.min(history['loss']))[0][0]
min_val_loss_idx 										= np.where(history['val_loss']==np.min(history['val_loss']))[0][0]
min_test_loss_idx 									= np.where(history['test_loss']==np.min(history['test_loss']))[0][0]
max_train_accuracy_idx 							= np.where(history['binary_accuracy']==np.max(history['binary_accuracy']))[0][0]
max_val_accuracy_idx 								= np.where(history['val_binary_accuracy']==np.max(history['val_binary_accuracy']))[0][0]
max_test_accuracy_idx 							= np.where(history['test_binary_accuracy']==np.max(history['test_binary_accuracy']))[0][0]


#  Plot losses and accuracies
history_df 													= pd.DataFrame(columns=['epoch', 'loss', 'type'])

epochs 															= np.linspace(1, len(history['loss']), len(history['loss']), dtype=int)

history_df['epoch'] 								= np.concatenate((epochs, epochs, epochs))
history_df['loss'] 									= np.concatenate((history['loss'], history['val_loss'], history['test_loss']))
history_df['type'] 									= np.concatenate((["Train " + str('%.3f'%(history['loss'][min_train_loss_idx])) + "@" + str((min_train_loss_idx + 1)) for x in epochs], \
																											["Validation " + str('%.3f'%(history['val_loss'][min_val_loss_idx])) + "@" + str((min_val_loss_idx + 1)) for x in epochs], \
																											["Test " + str('%.3f'%(history['test_loss'][min_test_loss_idx])) + "@" + str((min_test_loss_idx + 1)) for x in epochs]))


my_plot_history_loss 								= (ggplot(history_df) \
																				+ aes(x='epoch', y = 'loss', color = 'type') \
																				+ geom_line() \
																				+ labs(title = "Loss", x = 'epoch', y = 'loss', color = 'Type')) \
																				+ scale_color_manual(values=['#FF0000', '#0000FF', '#FF7518']) \
																				+ ylim(ylims_loss[0], ylims_loss[1]) \
																				+ theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold"))
	
my_plot_history_loss.save(dir_name + '/loss', dpi=600)


history_df 													= pd.DataFrame(columns=['epoch', 'accuracy', 'type'])

history_df['epoch'] 								= np.concatenate((epochs, epochs, epochs))
history_df['accuracy'] 							= np.concatenate((history['binary_accuracy'], history['val_binary_accuracy'], history['test_binary_accuracy']))
history_df['type'] 									= np.concatenate((["Train " + str('%.3f'%(history['binary_accuracy'][max_train_accuracy_idx])) + "@" + str((max_train_accuracy_idx + 1)) for x in epochs], \
																											["Validation " + str('%.3f'%(history['val_binary_accuracy'][max_val_accuracy_idx])) + "@" + str((max_val_accuracy_idx + 1)) for x in epochs], \
																											["Test " + str('%.3f'%(history['test_binary_accuracy'][max_test_accuracy_idx])) + "@" + str((max_test_accuracy_idx + 1)) for x in epochs]))

my_plot_history_accuracy 						= (ggplot(history_df) \
																				+ aes(x='epoch', y = 'accuracy', color = 'type') \
																				+ geom_line() \
																				+ labs(title = "Accuracy", x = 'epoch', y = 'accuracy', color = 'Type')) \
																				+ scale_color_manual(values=['#FF0000', '#0000FF', '#FF7518']) \
																				+ ylim(ylims_acc[0], ylims_acc[1]) \
																				+ theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold"))
	
my_plot_history_accuracy.save(dir_name + '/accuracy', dpi=600)

num_classes_iterations 							= args.num_classes-1 if args.num_classes == 2 and args.out_act_fun_2_classes == "sigmoid" else args.num_classes

#  Plot ROC curves
fpr 																= dict()
tpr 																= dict()
roc_auc 														= dict()
roc_df 															= pd.DataFrame(columns=['fpr', 'tpr', 'type'])

for i in range(num_classes_iterations):
  	fpr[i], tpr[i], _ 							= roc_curve(y_test[:, i] if args.num_classes != 2 else y_test, y_pred[:, i] if args.num_classes != 2 else y_pred)
  	roc_auc[i] 											= auc(fpr[i], tpr[i])
  	print("AUC class {0}: {1:0.3f}".format(classes_dic[i], roc_auc[i]))

 if args.num_classes == 24:
 	print("Average AUC 5 classes:", np.sum(roc_auc[0:5]) / 5)
	print("Average AUC 20 classes:", (roc_auc[0] + np.sum(roc_auc[5:24])) / 20)

roc_df['fpr'] 											= np.concatenate(list(fpr.values()), axis=0)
roc_df['tpr'] 											= np.concatenate(list(tpr.values()), axis=0)
roc_df['type'] 											= np.concatenate([['{0} (area = {1:0.2f})'.format(classes_dic.get(i), roc_auc[i]) for x in range(0, len(fpr[i]))] for i in range(num_classes_iterations)])

roc_df.type 												= pd.Categorical(roc_df.type, \
																				ordered 		= True, \
																				categories 	= ['{0} (area = {1:0.2f})'.format(classes_dic.get(i), roc_auc[i]) for i in range(num_classes_iterations)])

my_plot_roc 												= (ggplot(roc_df) \
																				+ aes(x='fpr', y = 'tpr', color = 'type') \
																				+ geom_line() \
																				+ geom_abline(linetype="dotted") \
																				+ scale_colour_discrete() \
																				+ xlim(0, 1) \
																				+ ylim(0, 1) \
																				+ labs(title = "ROC curves", x = 'False Positive Rate', y = 'True Positive Rate', color = 'Class') \
																				+ theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold")))

my_plot_roc.save(dir_name + '/roc', dpi=600)


#  Plot the PR curves
lr_precision 													= dict()
lr_recall 														= dict()
pr_auc 																= dict()
pr_df 																= pd.DataFrame(columns=['lr_precision', 'lr_recall', 'type'])

for i in range(num_classes_iterations):
  	lr_precision[i], lr_recall[i], _ 	= precision_recall_curve(y_test[:, i] if args.num_classes != 2 else y_test, y_pred[:, i] if args.num_classes != 2 else y_pred)
  	pr_auc[i] 												= auc(lr_recall[i], lr_precision[i])

pr_df['lr_precision'] 								= np.concatenate(list(lr_precision.values()), axis=0)
pr_df['lr_recall'] 										= np.concatenate(list(lr_recall.values()), axis=0)
pr_df['type'] 												= np.concatenate([['{0} (area = {1:0.2f})'.format(classes_dic.get(i), pr_auc[i]) for x in range(0, len(lr_precision[i]))] for i in range(num_classes_iterations)])
    
pr_df.type 														= pd.Categorical(pr_df.type, \
																					ordered 		= True, \
																					categories 	= ['{0} (area = {1:0.2f})'.format(classes_dic.get(i), pr_auc[i]) for i in range(num_classes_iterations)])

my_plot_pr 														= (ggplot(pr_df) \
																					+ aes(x='lr_precision', y = 'lr_recall', color = 'type') \
																					+ geom_line() \
																					+ geom_abline(slope=-1, intercept=1, linetype="dotted") \
																					+ scale_colour_discrete() \
																					+ xlim(0, 1) \
																					+ ylim(0, 1) \
																					+ labs(title = "PR curves", x = 'Recall', y = 'Precision', color = 'Type') \
																					+ theme(plot_title = element_text(face="bold"), axis_title_x  = element_text(face="bold"), axis_title_y = element_text(face="bold"), legend_title = element_text(face="bold")))

my_plot_pr.save(dir_name + '/pr', dpi=600)

#  Generate and print the confusion matrices
y_pred 																= np.where(y_pred > args.sigmoid_threshold, 1, 0)

cm 																		= multilabel_confusion_matrix(y_test, y_pred)

for i in range(0, len(cm)):
  print("Confusion matrix " + classes_dic.get(i) + ":\n", pd.DataFrame(cm[i]))
  make_confusion_matrix(cm[i], figsize=(10, 8), cbar=False, sum_stats=False, title=" confusion matrix", type=classes_dic.get(i), dir=dir_name)

print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))
print("\n\n")


#  Print the metrics on the test set
avg 																= 'macro'

if args.num_classes == 2:
	avg 															= 'micro'

accuracy 														= dict()

for i in range(num_classes_iterations):
	accuracy[i] 											= accuracy_score(y_test[:, i] if args.num_classes != 2 else y_test, y_pred[:, i] if args.num_classes != 2 else y_pred)
	print("Accuracy class {0}: {1:0.2f}".format(classes_dic[i], accuracy[i]))

if args.num_classes == 24:
	print("Average accuracy 5 classes:", np.sum(accuracy[0:5]) / 5)
	print("Average accuracy 20 classes:", (accuracy[0] + np.sum(accuracy[5:24])) / 20)

acc 																= np.mean(list(accuracy.values()))
roc 																= np.mean(list(roc_auc.values()))
auprc 															= np.mean(list(pr_auc.values()))
f1 																	= f1_score(y_test, y_pred, average=avg)
prec 																= precision_score(y_test, y_pred, average=avg)
rec 																= recall_score(y_test, y_pred, average=avg)

print("Test accuracy:", acc)
print("Test AUC:", roc)
print("Test AUPRC:", auprc)
print("Test F1 score:", f1)
print("Test precision:", prec)
print("Test recall:", rec)