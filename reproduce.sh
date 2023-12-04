#!/bin/bash

if [[ ! -d ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ]]; then
	wget -r -N -c -np --quiet -O ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
fi

./buildme.sh

docker run -it --user $UID:$UID --rm --gpus all --runtime nvidia -v $(pwd):/home/docker/cnn-ecg danielebaccega/reproduce-cnn-ecg /usr/bin/bash -c "/home/docker/cnn-ecg/run_configurations.sh"