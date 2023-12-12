#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate tf-gpu

if [[ ! -d "TrainedModels" ]]; then
	mkdir TrainedModels
fi

# Reproduce the paper results
bash Configurations/D1.sh
bash Configurations/D1-D2.sh
bash Configurations/D1-V1.sh
bash Configurations/D1-V2.sh
bash Configurations/D1-V3.sh
bash Configurations/D1-V4.sh
bash Configurations/D1-V5.sh
bash Configurations/D1-V6.sh
bash Configurations/8leads.sh
bash Configurations/12leads.sh
bash Configurations/12leads_WithoutDataAugmentation.sh

python CI.py

# Test the CNN on the Georgia dataset (https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database)
bash GeorgiaRefinementLastLayer.sh
bash GeorgiaRefinementAll.sh