FROM continuumio/miniconda3

COPY conda_env.yml /

RUN conda env create -f /conda_env.yml
RUN rm /conda_env.yml
RUN conda init bash

WORKDIR /root




