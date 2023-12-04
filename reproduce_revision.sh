#!/bin/bash

if [[ ! -d ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ]]; then
	wget -r -N -c -np --quiet -O Georgia.zip https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database/download?datasetVersionNumber=1
	unzip Georgia.zip
	rm Georgia.zip
fi

./buildme.sh

docker run -it --user $UID:$UID --rm --gpus all --runtime nvidia -v $(pwd):/home/docker/cnn-ecg danielebaccega/reproduce-cnn-ecg /usr/bin/bash -c "/home/docker/cnn-ecg/run_configurations.sh"