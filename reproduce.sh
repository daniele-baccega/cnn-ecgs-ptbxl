#!/bin/bash

if [[ ! -d ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ]]; then
	wget -r -N -c -np --quiet -O ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
fi

docker build -t danielebaccega/reproduce-cnn-ecg .
docker run -it --user 1001:1001 -v $(pwd):/home/docker/cnn-ecg danielebaccega/reproduce-cnn-ecg /usr/bin/bash -c "/home/docker/cnn-ecg/run_configurations.sh"