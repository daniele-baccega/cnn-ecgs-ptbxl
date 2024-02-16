# Base image https://hub.docker.com/_/ubuntu
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
LABEL maintainer="Daniele Baccega <daniele.baccega@unito.it>"

RUN apt update \
    && apt install -y build-essential \
    && apt install -y wget \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh \
    && bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

COPY tf-gpu.yml tf-gpu.yml

RUN conda env create -f tf-gpu.yml

# Create cnn-ecg directory
RUN mkdir /home/docker; chmod -R 777 /home/docker
RUN mkdir /home/docker/cnn-ecg; chmod 777 /home/docker/cnn-ecg

WORKDIR /home/docker/cnn-ecg/

## Copy files
COPY main.py .
COPY model.py .
COPY datagenerator.py .
COPY utils.py .
COPY run.sh .
COPY run_configurations.sh .
COPY reproduce.sh .
COPY Configurations .
COPY GeorgiaRefinementLastLayer.sh .
COPY GeorgiaRefinementAll.sh .
COPY ChinaRefinementLastLayer.sh .
COPY ChinaRefinementAll.sh .
COPY GeorgiaRefinementLastLayer.py .
COPY GeorgiaRefinementAll.py .
COPY ChinaRefinementLastLayer.py .
COPY ChinaRefinementAll.py .
COPY physionet_challenge_utility_script.py .
COPY SNOMED_mappings_scored_Georgia.csv .
COPY SNOMED_mappings_unscored_Georgia.csv .
COPY SNOMED_mappings_scored_China.csv .
COPY SNOMED_mappings_unscored_China.csv .
COPY CI.py .
COPY CI_GeorgiaRefinementLastLayer.py .
COPY CI_GeorgiaRefinementAll.py .
COPY CI_ChinaRefinementLastLayer.py .
COPY CI_ChinaRefinementAll.py .
COPY evaluate_model.py .
COPY helper_code.py .
COPY weights_abbreviations_Georgia.csv .
COPY weights_abbreviations_China.csv .
