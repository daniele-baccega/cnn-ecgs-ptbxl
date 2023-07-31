# Base image https://hub.docker.com/u/danielebaccega
FROM danielebaccega/cnn-ecg
LABEL maintainer="Daniele Baccega <daniele.baccega@unito.it>"

## Copy files
COPY main.py .
COPY model.py .
COPY datagenerator.py .
COPY utils.py .
COPY CI.py .
COPY run.sh .
COPY run_configurations.sh .
COPY reproduce.sh .
COPY Configurations .