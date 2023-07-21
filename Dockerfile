# Base image https://hub.docker.com/u/danielebaccega
FROM danielebaccega/cnn-ecg
LABEL maintainer="Daniele Baccega <daniele.baccega@unito.it>"

## Copy files
COPY main.py /home/docker/cnn-ecg/main.py
COPY model.py /home/docker/cnn-ecg/model.py
COPY datagenerator.py /home/docker/cnn-ecg/datagenerator.py
COPY utils.py /home/docker/cnn-ecg/utils.py
COPY CI.py /home/docker/cnn-ecg/CI.py
COPY run.sh /home/docker/cnn-ecg/run.sh
COPY run_configurations.sh /home/docker/cnn-ecg/run_configurations.sh
COPY reproduce.sh /home/docker/cnn-ecg/reproduce.sh
COPY Configurations /home/docker/cnn-ecg/Configurations