#!/bin/bash

if [[ ! -d ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ]]; then
	wget -r -N -c -np --quiet -O ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
	rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
fi

# Read this to correctly download the Georgia dataset using kaggle command: https://medium.com/@c.venkataramanan1/setting-up-kaggle-api-in-linux-b05332cde53a
if [[ ! -d "Georgia" ]]; then
        kaggle datasets download -d bjoernjostein/georgia-12lead-ecg-challenge-database
        unzip -qq georgia-12lead-ecg-challenge-database.zip
        rm georgia-12lead-ecg-challenge-database.zip
        mv WFDB Georgia
fi

./buildme.sh

docker run -it --user $UID:$UID --rm --gpus all --runtime nvidia -v $(pwd):/home/docker/cnn-ecg danielebaccega/reproduce-cnn-ecg /usr/bin/bash -c "/home/docker/cnn-ecg/run_configurations.sh"
