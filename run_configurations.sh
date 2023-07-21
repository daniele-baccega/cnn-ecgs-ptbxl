#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate tf-gpu

if [[ ! -d "/home/docker/cnn-ecg/TrainedModels" ]]; then
	mkdir /home/docker/cnn-ecg/TrainedModels
fi

# Reproduce the paper results
bash /home/docker/cnn-ecg/Configurations/D1.sh
bash /home/docker/cnn-ecg/Configurations/D1-D2.sh
bash /home/docker/cnn-ecg/Configurations/D1-V1.sh
bash /home/docker/cnn-ecg/Configurations/D1-V2.sh
bash /home/docker/cnn-ecg/Configurations/D1-V3.sh
bash /home/docker/cnn-ecg/Configurations/D1-V4.sh
bash /home/docker/cnn-ecg/Configurations/D1-V5.sh
bash /home/docker/cnn-ecg/Configurations/D1-V6.sh
bash /home/docker/cnn-ecg/Configurations/8leads.sh
bash /home/docker/cnn-ecg/Configurations/12leads.sh
bash /home/docker/cnn-ecg/Configurations/12leads_WithoutDataAugmentation.sh

python /home/docker/cnn-ecg/CI.py