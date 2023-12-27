#!/bin/bash

: '
	Deep learning project (100Hz)

	run_all.sh

	Runs the binary classification, the five-class multi-label classification, the twenty-class multi-label classification and
	generates the plots from the obtained results.

	Authors: Daniele Baccega, Andrea Saglietto
	Topic: Deep Learning applied to ECGs
	Dataset: https://physionet.org/content/ptb-xl/1.0.1/
	Description: The PTB-XL ECG dataset is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length
	where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22).
	The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record.
	The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements.
	To ensure comparability of machine learning algorithms trained on the dataset, we provide recommended splits into training and test sets.
'

S=0
NUM_CLASSES="20"
DATASET="../../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
EPOCHS="200"
BATCH_SIZE="32"
INIT_LEARNING_RATE="1e-2"
FINAL_LEARNING_RATE="1e-4"
LEADS="I V1"
CROP_WINDOW="344"
PADDING="0"
TIME_SCALE="0.8 1.2"
AMPLITUDE_SCALE="0.7 1.3"
OPTIMIZER="Adam"
MOMENTUM="0.9"
DROPOUT="0"
POOL_SIZES="2 2 2 2 2 2"
DILATION_FACTORS="2 2 2 2 2 2"
KERNEL_SIZE_LAST_CONV="1"
OUT_ACT_FUN_2_CLASSES="sigmoid"
RPEAK=""
DIFFERENT_FILTERS=""
NUM_FILTERS_FIRST_CONV="1"
ONE_D_MODEL=""
JITTER_STD="0.01 0.1"

DIRECTORY="D1-V1/"
mkdir $DIRECTORY

RUN=50

rm -fr data

bash run.sh --seed $S --num_classes $NUM_CLASSES --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH_SIZE --init_learning_rate $INIT_LEARNING_RATE \
         --final_learning_rate $FINAL_LEARNING_RATE --leads $LEADS --crop_window $CROP_WINDOW --padding $PADDING --time_scale $TIME_SCALE \
         --amplitude_scale $AMPLITUDE_SCALE --optimizer $OPTIMIZER --momentum $MOMENTUM --dropout $DROPOUT --pool_sizes $POOL_SIZES \
         --dilation_factors $DILATION_FACTORS --kernel_size_last_conv $KERNEL_SIZE_LAST_CONV --out_act_fun_2_classes $OUT_ACT_FUN_2_CLASSES \
         --directory "${DIRECTORY}20Classes_${S}" --num_filters_first_conv $NUM_FILTERS_FIRST_CONV --jitter_std $JITTER_STD $RPEAK $DIFFERENT_FILTERS $ONE_D_MODEL

S=$((S+1))

while [ $S -lt $RUN ]
do
        bash run.sh --seed $S --num_classes $NUM_CLASSES --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH_SIZE --init_learning_rate $INIT_LEARNING_RATE \
                 --final_learning_rate $FINAL_LEARNING_RATE --leads $LEADS --crop_window $CROP_WINDOW --padding $PADDING --time_scale $TIME_SCALE \
                 --amplitude_scale $AMPLITUDE_SCALE --optimizer $OPTIMIZER --momentum $MOMENTUM --dropout $DROPOUT --pool_sizes $POOL_SIZES \
                 --dilation_factors $DILATION_FACTORS --kernel_size_last_conv $KERNEL_SIZE_LAST_CONV --out_act_fun_2_classes $OUT_ACT_FUN_2_CLASSES \
                 --directory "${DIRECTORY}20Classes_${S}" --num_filters_first_conv $NUM_FILTERS_FIRST_CONV --jitter_std $JITTER_STD $RPEAK $DIFFERENT_FILTERS $ONE_D_MODEL -d

        S=$((S+1))
done

mv $DIRECTORY TrainedModels
