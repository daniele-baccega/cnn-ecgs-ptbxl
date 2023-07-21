#!/bin/bash

: '
  Deep learning project (100Hz)

  run.sh

  Runs a specific classification task (on two, five or twenty classes).

  Inputs:
   	  -dir  or  --directory:   	          output directory
      -d    or  --data:                   use the saved data (previously generated with another run); don''t use this flag if, for example, you
                                          changed the train, validation and/or test sets
      -s    or  --seed:                   random seed
  	  -nc   or  --num_classes: 	          specific classification task (binary classification, five-class multi-label classification or
  							 	                        twenty-class multi-label classification)
  	  -ds   or  --dataset:     	          path to the dataset
  	  -e    or  --epochs:      	          number of epochs
      -bs   or  --batch_size:  	          batch size
      -ilr  or  --init_learning_rate: 	  initial learning rate
      -flr  or  --final_learning_rate:    final learning rate
      -l    or  --leads:                  leads to use
      -cw   or  --crop_window:            dimension of the window used for the cropping
      -p    or  --padding:                number of zeros to add before and after each cropped lead
      -ts   or  --time_scale:             interval used to alter a bit the frequency of the ECGs
      -as   or  --amplitude_scale:        interval used to alter a bit the voltage of the ECGs
      -o    or  --optimizer:              optimizer
      -m    or  --momentum:               momentum for SGD optimizer
      -dr   or  --dropout:                spatial dropout before the last convolutional layer
      -ps   or  --pool_sizes:             size of the six max pooling layers
      -df   or  --dilation_factors:       dilation factors of the six convolutional 'temporal' layers
      -kslc or  --kernel_size_last_conv:  kernel size of the last convolutional 'spatial' layer
      -oaf2 or  --out_act_fun_2_classes:  output activation function for the two classes classification task

  Authors: Daniele Baccega, Andrea Saglietto
  Topic: Deep Learning applied to ECGs
  Dataset: https://physionet.org/content/ptb-xl/1.0.1/
  Description: The PTB-XL ECG dataset is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length
  where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22).
  The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record.
  The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements.
  To ensure comparability of machine learning algorithms trained on the dataset, we provide recommended splits into training and test sets.
'

SEED="0"
NUM_CLASSES="2"
DATASET="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
EPOCHS="200"
BATCH_SIZE="32"
INIT_LEARNING_RATE="1e-2"
FINAL_LEARNING_RATE="1e-4"
LEADS="I II V1 V2 V3 V4 V5 V6"
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

DIR_NAME="${NUM_CLASSES}Classes_$(date +%s)"
USE_SAVED_DATA=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--seed)
      SEED="$2"
      shift
      shift
      ;;
      -nc|--num_classes)
      NUM_CLASSES="$2"
      shift
      shift
      ;;
    -ds|--dataset)
      DATASET="$2"
      shift
      shift
      ;;
    -e|--epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    -bs|--batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    -ilr|--init_learning_rate)
      INIT_LEARNING_RATE="$2"
      shift
      shift
      ;;
    -flr|--final_learning_rate)
      FINAL_LEARNING_RATE="$2"
      shift
      shift
      ;;
    -l|--leads)
      LEADS="$2"
      shift
      shift
      while [[ $1 != -* ]] && [[ $1 != "" ]]; do
        LEADS="$LEADS $1"
        shift
      done
      ;;
    -cw|--crop_window)
      CROP_WINDOW="$2"
      shift
      shift
      ;;
    -p|--padding)
      PADDING="$2"
      shift
      shift
      ;;
    -ts|--time_scale)
      TIME_SCALE="$2 $3"
      shift
      shift
      shift
      ;;
    -as|--amplitude_scale)
      AMPLITUDE_SCALE="$2 $3"
      shift
      shift
      shift
      ;;
    -o|--optimizer)
      OPTIMIZER="$2"
      shift
      shift
      ;;
    -m|--momentum)
      MOMENTUM="$2"
      shift
      shift
      ;;
    -dr|--dropout)
      DROPOUT="$2"
      shift
      shift
      ;;
    -ps|--pool_sizes)
      POOL_SIZES="$2"
      shift
      shift
      while [[ $1 != -* ]] && [[ $1 != "" ]]; do
        POOL_SIZES="$POOL_SIZES $1"
        shift
      done
      ;;
    -df|--dilation_factors)
      DILATION_FACTORS="$2"
      shift
      shift
      while [[ $1 != -* ]] && [[ $1 != "" ]]; do
        DILATION_FACTORS="$DILATION_FACTORS $1"
        shift
      done
      ;;
    -kslc|--kernel_size_last_conv)
      KERNEL_SIZE_LAST_CONV="$2"
      shift
      shift
      ;;
    -oaf2|--out_act_fun_2_classes)
      OUT_ACT_FUN_2_CLASSES="$2"
      shift
      shift
      ;;
    -dir|--directory)
      DIR_NAME="$2"
      shift
      shift
      ;;
    -d|--data)
      USE_SAVED_DATA=1
      shift
      shift
      ;;
    -h|--help)
  	  printf "./run.sh - runs a specific classification task (on two, five or twenty classes)\n\n"
  	  printf "Arguments:\n"
      printf "        -dir  or  --directory:               output directory\n"
      printf "        -d    or  --data:                    use the saved data (previously generated with another run); don't use this flag if, for example, you\n"
      printf "                                             changed the train, validation and/or test sets\n"
      printf "        -s    or  --seed:                    random seed\n"
      printf "        -nc   or  --num_classes:             specific classification task (binary classification, five-class multi-label classification or\n"
  	  printf "                                             twenty-class multi-label classification)\n"
  	  printf "        -ds   or  --dataset:                 path to the dataset\n"
  	  printf "        -e    or  --epochs:                  number of epochs\n"
  	  printf "        -bs   or  --batch_size:              batch size\n"
  	  printf "        -ilr  or  --init_learning_rate:      initial learning rate\n"
      printf "        -flr  or  --final_learning_rate:     initial learning rate\n"
      printf "        -l    or  --leads:                   leads to use\n"
      printf "        -cw   or  --crop_window:             dimension of the window used for the cropping\n"
      printf "        -p    or  --padding:                 number of zeros to add before and after each cropped lead\n"
      printf "        -ts   or  --time_scale:              interval used to alter a bit the frequency of the ECGs\n"
      printf "        -as   or  --amplitude_scale:         interval used to alter a bit the voltage of the ECGs\n"
      printf "        -o    or  --optimizer:               optimizer\n"
      printf "        -m    or  --momentum:                momentum for SGD optimizer\n"
      printf "        -dr   or  --dropout:                 spatial dropout before the last convolutional layer\n"
      printf "        -ps   or  --pool_sizes:              size of the six max pooling layers\n"
      printf "        -df   or  --dilation_factors:        dilation factors of the six convolutional 'temporal' layers\n"
      printf "        -kslc or  --kernel_size_last_conv:   kernel size of the last convolutional 'spatial' layer\n"
      printf "        -oaf2 or  --out_act_fun_2_classes:   output activation function for the two classes classification task\n"
      exit 1
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ $USE_SAVED_DATA -eq 0 ] && [ -d "data/${NUM_CLASSES}Classes" ]; then
	rm -r "data/${NUM_CLASSES}Classes"
fi

if [ $NUM_CLASSES -eq 2 ] || [ $NUM_CLASSES -eq 5 ] || [ $NUM_CLASSES -eq 20 ]; then
	if ! [ -d "$DIR_NAME" ]; then
		mkdir $DIR_NAME

		cp main.py $DIR_NAME
		cp model.py $DIR_NAME
		cp utils.py $DIR_NAME
		cp datagenerator.py $DIR_NAME
		cp explainability.py $DIR_NAME

    if [ ! -d "data" ]; then
      mkdir "data"
    fi

		cd $DIR_NAME

    mkdir checkpoints
		
		python3 main.py --seed $SEED --num_classes $NUM_CLASSES --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH_SIZE --init_learning_rate $INIT_LEARNING_RATE \
                    --final_learning_rate $FINAL_LEARNING_RATE --leads $LEADS --crop_window $CROP_WINDOW --padding $PADDING --time_scale $TIME_SCALE \
                    --amplitude_scale $AMPLITUDE_SCALE --optimizer $OPTIMIZER --momentum $MOMENTUM --dropout $DROPOUT --pool_sizes $POOL_SIZES \
                    --dilation_factors $DILATION_FACTORS --kernel_size_last_conv $KERNEL_SIZE_LAST_CONV --out_act_fun_2_classes $OUT_ACT_FUN_2_CLASSES
	  
    cd ..
    
    python3 process_results.py --num_classes $NUM_CLASSES --out_act_fun_2_classes $OUT_ACT_FUN_2_CLASSES
  fi
else
	echo "The number of classes to use must be equals to 2, 5 or 20."
  exit 1
fi